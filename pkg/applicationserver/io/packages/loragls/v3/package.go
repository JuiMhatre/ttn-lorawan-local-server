// Copyright © 2021 The Things Network Foundation, The Things Industries B.V.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package loracloudgeolocationv3 enables LoRa Cloud Geolocation Services integration.
package loracloudgeolocationv3

import (
	"context"
	"fmt"
	"time"

	apppayload "go.thethings.network/lorawan-application-payload"
	"go.thethings.network/lorawan-stack/v3/pkg/applicationserver/io"
	"go.thethings.network/lorawan-stack/v3/pkg/applicationserver/io/packages"
	"go.thethings.network/lorawan-stack/v3/pkg/applicationserver/io/packages/loragls/v3/api"
	"go.thethings.network/lorawan-stack/v3/pkg/errors"
	"go.thethings.network/lorawan-stack/v3/pkg/events"
	"go.thethings.network/lorawan-stack/v3/pkg/goproto"
	"go.thethings.network/lorawan-stack/v3/pkg/jsonpb"
	"go.thethings.network/lorawan-stack/v3/pkg/log"
	"go.thethings.network/lorawan-stack/v3/pkg/ttnpb"
	urlutil "go.thethings.network/lorawan-stack/v3/pkg/util/url"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// PackageName defines the package name.
const PackageName = "lora-cloud-geolocation-v3"

// GeolocationPackage is the LoRa Cloud Geolocation application package.
type GeolocationPackage struct {
	server   io.Server
	registry packages.Registry
}

var (
	errLocationQuery = errors.DefineInternal("location_query", "location query")
	errNoResult      = errors.DefineNotFound("no_result", "no location query result")
	errNoAssociation = errors.DefineInternal("no_association", "no association available")
)

// HandleUp implements packages.ApplicationPackageHandler.
func (p *GeolocationPackage) HandleUp(
	ctx context.Context,
	def *ttnpb.ApplicationPackageDefaultAssociation,
	assoc *ttnpb.ApplicationPackageAssociation,
	up *ttnpb.ApplicationUp,
) (err error) {
	ctx = log.NewContextWithField(ctx, "namespace", "applicationserver/io/packages/loragls/v1")
	ctx = events.ContextWithCorrelationID(
		ctx, append(
			up.CorrelationIds,
			fmt.Sprintf("as:packages:loracloudglsv3:%s", events.NewCorrelationID()),
		)...,
	)

	if def == nil && assoc == nil {
		return errNoAssociation.New()
	}

	defer func() {
		if err != nil {
			registerPackageFail(ctx, up.EndDeviceIds, err)
		}
	}()

	data, err := p.mergePackageData(def, assoc)
	if err != nil {
		return err
	}

	switch m := up.Up.(type) {
	case *ttnpb.ApplicationUp_UplinkMessage:
		pkgAssocIDs := assocIDs(def, assoc, up.EndDeviceIds)
		assoc, err := p.pushUplink(ctx, pkgAssocIDs, m.UplinkMessage, data)
		if err != nil {
			return err
		}

		data, err = p.mergePackageData(def, assoc)
		if err != nil {
			return err
		}

		return p.sendQuery(ctx, up.EndDeviceIds, m.UplinkMessage, data)
	default:
		return nil
	}
}

// Package implements packages.ApplicationPackageHandler.
func (p *GeolocationPackage) Package() *ttnpb.ApplicationPackage {
	return &ttnpb.ApplicationPackage{
		Name:         PackageName,
		DefaultFPort: 197,
	}
}

// assocIDs returns the identifiers of the given association. If the association is nil, new identifiers are created.
func assocIDs(
	def *ttnpb.ApplicationPackageDefaultAssociation,
	assoc *ttnpb.ApplicationPackageAssociation,
	ids *ttnpb.EndDeviceIdentifiers,
) *ttnpb.ApplicationPackageAssociationIdentifiers {
	assocIDs := assoc.GetIds()
	if assocIDs == nil {
		assocIDs = &ttnpb.ApplicationPackageAssociationIdentifiers{
			EndDeviceIds: ids,
			FPort:        def.Ids.FPort,
		}
	}
	return assocIDs
}

// New instantiates the LoRa Cloud Geolocation package.
func New(server io.Server, registry packages.Registry) packages.ApplicationPackageHandler {
	return &GeolocationPackage{
		server:   server,
		registry: registry,
	}
}

func (p *GeolocationPackage) singleFrameQuery(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, up *ttnpb.ApplicationUplink, data *Data, client *api.Client) (api.AbstractLocationSolverResponse, error) {
	mds, err := RxMDSliceFromProto(up.RxMetadata)
	if err != nil {
		return nil, err
	}
	req := api.BuildSingleFrameRequest(ctx, mds)
	if len(req.Gateways) < 1 {
		return nil, nil
	}
	resp, err := client.SolveSingleFrame(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.AbstractResponse(), nil
}

func (p *GeolocationPackage) gnssQuery(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, up *ttnpb.ApplicationUplink, data *Data, client *api.Client) (api.AbstractLocationSolverResponse, error) {
	req := &api.GNSSRequest{}
	if up.DecodedPayload == nil {
		req.Payload = up.FrmPayload
	} else {
		m, err := goproto.Map(up.DecodedPayload)
		if err != nil {
			return nil, err
		}
		payload, ok := apppayload.InferGNSS(m)
		if !ok {
			return nil, nil
		}
		req.Payload = payload
	}
	resp, err := client.SolveGNSS(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.AbstractResponse(), nil
}

func minInt(a int, b int) int {
	if a <= b {
		return a
	}
	return b
}

// pushUplink updates the package data with the given uplink.
func (p *GeolocationPackage) pushUplink(
	ctx context.Context,
	ids *ttnpb.ApplicationPackageAssociationIdentifiers,
	up *ttnpb.ApplicationUplink,
	data *Data,
) (*ttnpb.ApplicationPackageAssociation, error) {
	maxUplinkCount := minInt(data.MultiFrameWindowSize, 16)

	setter := func(assoc *ttnpb.ApplicationPackageAssociation) (*ttnpb.ApplicationPackageAssociation, []string, error) {
		data := &Data{}
		fieldMask := []string{"data"}

		if assoc == nil {
			assoc = &ttnpb.ApplicationPackageAssociation{
				Ids:         ids,
				PackageName: PackageName,
			}
			fieldMask = []string{"data", "ids", "package_name"}
		} else {
			if err := data.FromStruct(assoc.Data); err != nil {
				return nil, nil, err
			}
		}

		if maxUplinkCount != 0 && len(data.RecentMetadata) >= maxUplinkCount {
			data.RecentMetadata = data.RecentMetadata[1:]
		}

		md := &UplinkMetadata{}
		if err := md.FromApplicationUplink(up); err != nil {
			return nil, nil, err
		}

		data.RecentMetadata = append(data.RecentMetadata, md)
		st, err := data.Struct()
		if err != nil {
			return nil, nil, err
		}
		assoc.Data = st

		return assoc, fieldMask, nil
	}

	return p.registry.SetAssociation(ctx, ids, []string{"data"}, setter)
}

func (*GeolocationPackage) multiFrameQuery(
	ctx context.Context,
	_ *ttnpb.EndDeviceIdentifiers,
	up *ttnpb.ApplicationUplink,
	data *Data,
	client *api.Client,
) (api.AbstractLocationSolverResponse, error) {
	count := data.MultiFrameWindowSize
	if count == 0 && len(up.FrmPayload) > 0 {
		count = int(up.FrmPayload[0])
		count = minInt(count, 16)
	}
	if count == 0 {
		return nil, nil
	}

	now := time.Now()
	mds := make([][]*api.RxMetadata, 0, count)

	for _, md := range data.RecentMetadata {
		if now.Sub(md.ReceivedAt) > data.MultiFrameWindowAge {
			continue
		}

		mds = append(mds, md.RxMetadata)

		if len(mds) >= count {
			break
		}
	}

	req := api.BuildMultiFrameRequest(ctx, mds)
	if len(req.Gateways) < 1 {
		return nil, nil
	}
	resp, err := client.SolveMultiFrame(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.AbstractResponse(), nil
}

func (p *GeolocationPackage) wifiQuery(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, up *ttnpb.ApplicationUplink, data *Data, client *api.Client) (api.AbstractLocationSolverResponse, error) {
	m, err := goproto.Map(up.DecodedPayload)
	if err != nil {
		return nil, err
	}
	aps, ok := apppayload.InferWiFiAccessPoints(m)
	if !ok {
		return nil, nil
	}
	if len(aps) < 2 {
		return nil, nil
	}
	accessPoints := make([]api.AccessPoint, 0, len(aps))
	for _, accessPoint := range aps {
		accessPoints = append(accessPoints, api.AccessPoint{
			MACAddress: fmt.Sprintf("%x:%x:%x:%x:%x:%x",
				accessPoint.BSSID[0],
				accessPoint.BSSID[1],
				accessPoint.BSSID[2],
				accessPoint.BSSID[3],
				accessPoint.BSSID[4],
				accessPoint.BSSID[5],
			),
			SignalStrength: int64(accessPoint.RSSI),
		})
	}
	mds, err := RxMDSliceFromProto(up.RxMetadata)
	if err != nil {
		return nil, err
	}
	req := api.BuildWiFiRequest(ctx, mds, accessPoints)
	if len(req.LoRaWAN) < 1 {
		return nil, nil
	}
	resp, err := client.SolveWiFi(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.AbstractResponse(), nil
}

func (p *GeolocationPackage) sendQuery(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, up *ttnpb.ApplicationUplink, data *Data) error {
	var runQuery func(context.Context, *ttnpb.EndDeviceIdentifiers, *ttnpb.ApplicationUplink, *Data, *api.Client) (api.AbstractLocationSolverResponse, error)
	switch data.Query {
	case QUERY_TOARSSI:
		if data.MultiFrame {
			runQuery = p.multiFrameQuery
		} else {
			runQuery = p.singleFrameQuery
		}
	case QUERY_GNSS:
		runQuery = p.gnssQuery
	case QUERY_TOAWIFI:
		runQuery = p.wifiQuery
	default:
		return nil
	}

	httpClient, err := p.server.HTTPClient(ctx)
	if err != nil {
		return err
	}
	client, err := api.New(httpClient, api.WithToken(data.Token), api.WithBaseURL(data.ServerURL))
	if err != nil {
		return err
	}

	resp, err := runQuery(ctx, ids, up, data, client)
	if err != nil || resp == nil {
		return err
	}

	resultStruct, err := toStruct(resp.Raw())
	if err != nil {
		return err
	}

	if err := p.sendServiceData(ctx, ids, resultStruct); err != nil {
		return err
	}

	if errors := resp.Errors(); len(errors) > 0 {
		var details []proto.Message
		for _, message := range errors {
			details = append(details, &ttnpb.ErrorDetails{
				Code:          uint32(codes.Unknown),
				MessageFormat: message,
			})
		}
		return errLocationQuery.WithDetails(details...)
	}

	result := resp.Result()
	if result == nil {
		return errNoResult.New()
	}

	if err := p.sendLocationSolved(ctx, ids, result); err != nil {
		return err
	}

	return nil
}

func (p *GeolocationPackage) sendServiceData(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, data *structpb.Struct) error {
	return p.server.Publish(ctx, &ttnpb.ApplicationUp{
		EndDeviceIds:   ids,
		CorrelationIds: events.CorrelationIDsFromContext(ctx),
		ReceivedAt:     timestamppb.Now(),
		Up: &ttnpb.ApplicationUp_ServiceData{
			ServiceData: &ttnpb.ApplicationServiceData{
				Data:    data,
				Service: PackageName,
			},
		},
	})
}

func (p *GeolocationPackage) sendLocationSolved(ctx context.Context, ids *ttnpb.EndDeviceIdentifiers, result api.AbstractLocationSolverResult) error {
	loc := result.Location()
	return p.server.Publish(ctx, &ttnpb.ApplicationUp{
		EndDeviceIds:   ids,
		CorrelationIds: events.CorrelationIDsFromContext(ctx),
		ReceivedAt:     timestamppb.Now(),
		Up: &ttnpb.ApplicationUp_LocationSolved{
			LocationSolved: &ttnpb.ApplicationLocation{
				Service:  PackageName,
				Location: loc,
			},
		},
	})
}

func (*GeolocationPackage) mergePackageData(
	def *ttnpb.ApplicationPackageDefaultAssociation,
	assoc *ttnpb.ApplicationPackageAssociation,
) (*Data, error) {
	var defaultData, associationData Data
	if def != nil {
		if err := defaultData.FromStruct(def.Data); err != nil {
			return nil, err
		}
	}
	if assoc != nil {
		if err := associationData.FromStruct(assoc.Data); err != nil {
			return nil, err
		}
	}

	merged := Data{
		Query:                defaultData.Query,
		MultiFrame:           defaultData.MultiFrame,
		MultiFrameWindowSize: defaultData.MultiFrameWindowSize,
		MultiFrameWindowAge:  defaultData.MultiFrameWindowAge,
		ServerURL:            defaultData.ServerURL,
		Token:                defaultData.Token,
		RecentMetadata:       defaultData.RecentMetadata,
	}

	if associationData.Query != 0 {
		merged.Query = associationData.Query
	}
	if associationData.ServerURL != nil {
		merged.ServerURL = urlutil.CloneURL(associationData.ServerURL)
	}
	if associationData.Token != "" {
		merged.Token = associationData.Token
	}
	if len(associationData.RecentMetadata) > 0 {
		merged.RecentMetadata = associationData.RecentMetadata
	}
	if merged.ServerURL == nil {
		merged.ServerURL = urlutil.CloneURL(api.DefaultServerURL)
	}
	return &merged, nil
}

func toStruct(i any) (*structpb.Struct, error) {
	b, err := jsonpb.TTN().Marshal(i)
	if err != nil {
		return nil, err
	}
	st := &structpb.Struct{}
	err = jsonpb.TTN().Unmarshal(b, st)
	if err != nil {
		return nil, err
	}
	return st, nil
}

// RxMDSliceFromProto converts a slice of RxMetadata from a protobuf representation.
func RxMDSliceFromProto(pb []*ttnpb.RxMetadata) ([]*api.RxMetadata, error) {
	md := make([]*api.RxMetadata, len(pb))
	for i, m := range pb {
		md[i] = &api.RxMetadata{}
		if err := md[i].FromProto(m); err != nil {
			return nil, err
		}
	}
	return md, nil
}
