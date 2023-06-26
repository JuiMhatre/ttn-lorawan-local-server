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

package deviceclaimingserver

import (
	"context"

	"go.thethings.network/lorawan-stack/v3/pkg/auth/rights"
	"go.thethings.network/lorawan-stack/v3/pkg/errors"
	"go.thethings.network/lorawan-stack/v3/pkg/rpcmetadata"
	"go.thethings.network/lorawan-stack/v3/pkg/ttnpb"
	"go.thethings.network/lorawan-stack/v3/pkg/types"
	"google.golang.org/protobuf/types/known/emptypb"
)

var (
	errParseQRCode          = errors.Define("parse_qr_code", "parse QR code failed")
	errQRCodeData           = errors.DefineInvalidArgument("qr_code_data", "invalid QR code data")
	errNoJoinEUI            = errors.DefineInvalidArgument("no_join_eui", "failed to extract JoinEUI from request")
	errNoEUI                = errors.DefineFailedPrecondition("no_eui", "DevEUI/JoinEUI not set for device")
	errMethodUnavailable    = errors.DefineUnimplemented("method_unavailable", "method unavailable")
	errClaimingNotSupported = errors.DefineAborted(
		"claiming_not_supported",
		"claiming not supported for JoinEUI `{eui}`",
	)
)

// endDeviceClaimingServer is the front facing entity for gRPC requests.
type endDeviceClaimingServer struct {
	ttnpb.UnimplementedEndDeviceClaimingServerServer

	DCS *DeviceClaimingServer
}

// Claim implements EndDeviceClaimingServer.
func (edcs *endDeviceClaimingServer) Claim(
	ctx context.Context,
	req *ttnpb.ClaimEndDeviceRequest,
) (*ttnpb.EndDeviceIdentifiers, error) {
	// Check that the collaborator has necessary rights before attempting to claim it on an upstream.
	// Since this is part of the create device flow,
	// we check that the collaborator has the rights to create devices in the application.
	targetAppID := req.GetTargetApplicationIds()
	if err := rights.RequireApplication(ctx, targetAppID,
		ttnpb.Right_RIGHT_APPLICATION_DEVICES_WRITE,
	); err != nil {
		return nil, err
	}

	var (
		joinEUI, devEUI         types.EUI64
		claimAuthenticationCode string
	)
	if authenticatedIDs := req.GetAuthenticatedIdentifiers(); authenticatedIDs != nil {
		joinEUI = types.MustEUI64(req.GetAuthenticatedIdentifiers().JoinEui).OrZero()
		devEUI = types.MustEUI64(req.GetAuthenticatedIdentifiers().DevEui).OrZero()
		claimAuthenticationCode = req.GetAuthenticatedIdentifiers().AuthenticationCode
	} else if qrCode := req.GetQrCode(); qrCode != nil {
		conn, err := edcs.DCS.GetPeerConn(ctx, ttnpb.ClusterRole_QR_CODE_GENERATOR, nil)
		if err != nil {
			return nil, err
		}
		qrg := ttnpb.NewEndDeviceQRCodeGeneratorClient(conn)
		callOpt, err := rpcmetadata.WithForwardedAuth(ctx, edcs.DCS.AllowInsecureForCredentials())
		if err != nil {
			return nil, err
		}
		data, err := qrg.Parse(ctx, &ttnpb.ParseEndDeviceQRCodeRequest{
			QrCode: qrCode,
		}, callOpt)
		if err != nil {
			return nil, errQRCodeData.WithCause(err)
		}
		dev := data.GetEndDeviceTemplate().GetEndDevice()
		if dev == nil {
			return nil, errParseQRCode.New()
		}
		joinEUI = types.MustEUI64(dev.GetIds().JoinEui).OrZero()
		devEUI = types.MustEUI64(dev.GetIds().DevEui).OrZero()
		claimAuthenticationCode = dev.ClaimAuthenticationCode.Value
	} else {
		return nil, errNoJoinEUI.New()
	}

	claimer := edcs.DCS.endDeviceClaimingUpstream.JoinEUIClaimer(ctx, joinEUI)
	if claimer == nil {
		return nil, errClaimingNotSupported.WithAttributes("eui", joinEUI)
	}

	err := claimer.Claim(ctx, joinEUI, devEUI, claimAuthenticationCode)
	if err != nil {
		return nil, err
	}

	// Echo identifiers from the request.
	return &ttnpb.EndDeviceIdentifiers{
		DeviceId:       req.TargetDeviceId,
		ApplicationIds: req.TargetApplicationIds,
		DevEui:         devEUI.Bytes(),
		JoinEui:        joinEUI.Bytes(),
	}, nil
}

// Unclaim implements EndDeviceClaimingServer.
func (edcs *endDeviceClaimingServer) Unclaim(
	ctx context.Context,
	in *ttnpb.EndDeviceIdentifiers,
) (*emptypb.Empty, error) {
	dev, err := edcs.getEndDevice(ctx, in, ttnpb.Right_RIGHT_APPLICATION_DEVICES_WRITE)
	if err != nil {
		return nil, err
	}
	ids := dev.GetIds()
	if ids.JoinEui == nil || ids.DevEui == nil {
		return nil, errNoEUI.New()
	}

	joinEUI := types.MustEUI64(ids.JoinEui).OrZero()
	claimer := edcs.DCS.endDeviceClaimingUpstream.JoinEUIClaimer(ctx, joinEUI)
	if claimer == nil {
		return nil, errClaimingNotSupported.WithAttributes("eui", joinEUI)
	}
	if err := claimer.Unclaim(ctx, ids); err != nil {
		return nil, err
	}
	return ttnpb.Empty, nil
}

// GetInfoByJoinEUI implements EndDeviceClaimingServer.
func (edcs *endDeviceClaimingServer) GetInfoByJoinEUI(
	ctx context.Context,
	in *ttnpb.GetInfoByJoinEUIRequest,
) (*ttnpb.GetInfoByJoinEUIResponse, error) {
	joinEUI := types.MustEUI64(in.JoinEui).OrZero()
	claimer := edcs.DCS.endDeviceClaimingUpstream.JoinEUIClaimer(ctx, joinEUI)
	return &ttnpb.GetInfoByJoinEUIResponse{
		JoinEui:          joinEUI.Bytes(),
		SupportsClaiming: claimer != nil,
	}, nil
}

// GetClaimStatus implements EndDeviceClaimingServer.
func (edcs *endDeviceClaimingServer) GetClaimStatus(
	ctx context.Context,
	in *ttnpb.EndDeviceIdentifiers,
) (*ttnpb.GetClaimStatusResponse, error) {
	dev, err := edcs.getEndDevice(ctx, in, ttnpb.Right_RIGHT_APPLICATION_DEVICES_READ)
	if err != nil {
		return nil, err
	}
	ids := dev.GetIds()
	if ids.JoinEui == nil || ids.DevEui == nil {
		return nil, errNoEUI.New()
	}

	joinEUI := types.MustEUI64(ids.JoinEui).OrZero()
	claimer := edcs.DCS.endDeviceClaimingUpstream.JoinEUIClaimer(ctx, joinEUI)
	if claimer == nil {
		return nil, errClaimingNotSupported.WithAttributes("eui", joinEUI)
	}
	return claimer.GetClaimStatus(ctx, ids)
}

func (edcs *endDeviceClaimingServer) getEndDevice(
	ctx context.Context,
	ids *ttnpb.EndDeviceIdentifiers,
	requiredRights ...ttnpb.Right,
) (*ttnpb.EndDevice, error) {
	if err := rights.RequireApplication(ctx, ids.GetApplicationIds(),
		requiredRights...,
	); err != nil {
		return nil, err
	}
	conn, err := edcs.DCS.GetPeerConn(ctx, ttnpb.ClusterRole_ENTITY_REGISTRY, nil)
	if err != nil {
		return nil, err
	}
	client, err := ttnpb.NewEndDeviceRegistryClient(conn), nil
	if err != nil {
		return nil, err
	}
	callOpt, err := rpcmetadata.WithForwardedAuth(ctx, edcs.DCS.AllowInsecureForCredentials())
	if err != nil {
		return nil, err
	}
	return client.Get(ctx, &ttnpb.GetEndDeviceRequest{
		EndDeviceIds: ids,
		FieldMask:    ttnpb.FieldMask("ids"),
	}, callOpt)
}
