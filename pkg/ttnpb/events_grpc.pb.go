// Copyright © 2019 The Things Network Foundation, The Things Industries B.V.
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

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v4.22.2
// source: ttn/lorawan/v3/events.proto

package ttnpb

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

const (
	Events_Stream_FullMethodName      = "/ttn.lorawan.v3.Events/Stream"
	Events_FindRelated_FullMethodName = "/ttn.lorawan.v3.Events/FindRelated"
)

// EventsClient is the client API for Events service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type EventsClient interface {
	// Stream live events, optionally with a tail of historical events (depending on server support and retention policy).
	// Events may arrive out-of-order.
	Stream(ctx context.Context, in *StreamEventsRequest, opts ...grpc.CallOption) (Events_StreamClient, error)
	FindRelated(ctx context.Context, in *FindRelatedEventsRequest, opts ...grpc.CallOption) (*FindRelatedEventsResponse, error)
}

type eventsClient struct {
	cc grpc.ClientConnInterface
}

func NewEventsClient(cc grpc.ClientConnInterface) EventsClient {
	return &eventsClient{cc}
}

func (c *eventsClient) Stream(ctx context.Context, in *StreamEventsRequest, opts ...grpc.CallOption) (Events_StreamClient, error) {
	stream, err := c.cc.NewStream(ctx, &Events_ServiceDesc.Streams[0], Events_Stream_FullMethodName, opts...)
	if err != nil {
		return nil, err
	}
	x := &eventsStreamClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type Events_StreamClient interface {
	Recv() (*Event, error)
	grpc.ClientStream
}

type eventsStreamClient struct {
	grpc.ClientStream
}

func (x *eventsStreamClient) Recv() (*Event, error) {
	m := new(Event)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *eventsClient) FindRelated(ctx context.Context, in *FindRelatedEventsRequest, opts ...grpc.CallOption) (*FindRelatedEventsResponse, error) {
	out := new(FindRelatedEventsResponse)
	err := c.cc.Invoke(ctx, Events_FindRelated_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// EventsServer is the server API for Events service.
// All implementations must embed UnimplementedEventsServer
// for forward compatibility
type EventsServer interface {
	// Stream live events, optionally with a tail of historical events (depending on server support and retention policy).
	// Events may arrive out-of-order.
	Stream(*StreamEventsRequest, Events_StreamServer) error
	FindRelated(context.Context, *FindRelatedEventsRequest) (*FindRelatedEventsResponse, error)
	mustEmbedUnimplementedEventsServer()
}

// UnimplementedEventsServer must be embedded to have forward compatible implementations.
type UnimplementedEventsServer struct {
}

func (UnimplementedEventsServer) Stream(*StreamEventsRequest, Events_StreamServer) error {
	return status.Errorf(codes.Unimplemented, "method Stream not implemented")
}
func (UnimplementedEventsServer) FindRelated(context.Context, *FindRelatedEventsRequest) (*FindRelatedEventsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method FindRelated not implemented")
}
func (UnimplementedEventsServer) mustEmbedUnimplementedEventsServer() {}

// UnsafeEventsServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to EventsServer will
// result in compilation errors.
type UnsafeEventsServer interface {
	mustEmbedUnimplementedEventsServer()
}

func RegisterEventsServer(s grpc.ServiceRegistrar, srv EventsServer) {
	s.RegisterService(&Events_ServiceDesc, srv)
}

func _Events_Stream_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(StreamEventsRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(EventsServer).Stream(m, &eventsStreamServer{stream})
}

type Events_StreamServer interface {
	Send(*Event) error
	grpc.ServerStream
}

type eventsStreamServer struct {
	grpc.ServerStream
}

func (x *eventsStreamServer) Send(m *Event) error {
	return x.ServerStream.SendMsg(m)
}

func _Events_FindRelated_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(FindRelatedEventsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EventsServer).FindRelated(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Events_FindRelated_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EventsServer).FindRelated(ctx, req.(*FindRelatedEventsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Events_ServiceDesc is the grpc.ServiceDesc for Events service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Events_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "ttn.lorawan.v3.Events",
	HandlerType: (*EventsServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "FindRelated",
			Handler:    _Events_FindRelated_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Stream",
			Handler:       _Events_Stream_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "ttn/lorawan/v3/events.proto",
}
