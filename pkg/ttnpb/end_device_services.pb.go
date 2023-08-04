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

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v4.22.2
// source: ttn/lorawan/v3/end_device_services.proto

package ttnpb

import (
	_ "google.golang.org/genproto/googleapis/api/annotations"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
	reflect "reflect"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

var File_ttn_lorawan_v3_end_device_services_proto protoreflect.FileDescriptor

var file_ttn_lorawan_v3_end_device_services_proto_rawDesc = []byte{
	0x0a, 0x28, 0x74, 0x74, 0x6e, 0x2f, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2f, 0x76, 0x33,
	0x2f, 0x65, 0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x73, 0x65, 0x72, 0x76,
	0x69, 0x63, 0x65, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0e, 0x74, 0x74, 0x6e, 0x2e,
	0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67,
	0x6c, 0x65, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x61, 0x6e, 0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1b, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65,
	0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x65, 0x6d, 0x70, 0x74, 0x79, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1f, 0x74, 0x74, 0x6e, 0x2f, 0x6c, 0x6f, 0x72, 0x61, 0x77,
	0x61, 0x6e, 0x2f, 0x76, 0x33, 0x2f, 0x65, 0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x20, 0x74, 0x74, 0x6e, 0x2f, 0x6c, 0x6f, 0x72, 0x61,
	0x77, 0x61, 0x6e, 0x2f, 0x76, 0x33, 0x2f, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x66, 0x69, 0x65,
	0x72, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x32, 0x9b, 0x08, 0x0a, 0x11, 0x45, 0x6e, 0x64,
	0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x52, 0x65, 0x67, 0x69, 0x73, 0x74, 0x72, 0x79, 0x12, 0x9d,
	0x01, 0x0a, 0x06, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x12, 0x26, 0x2e, 0x74, 0x74, 0x6e, 0x2e,
	0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x43, 0x72, 0x65, 0x61, 0x74,
	0x65, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x1a, 0x19, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e,
	0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x22, 0x50, 0x82, 0xd3,
	0xe4, 0x93, 0x02, 0x4a, 0x3a, 0x01, 0x2a, 0x22, 0x45, 0x2f, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63,
	0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x7b, 0x65, 0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69,
	0x63, 0x65, 0x2e, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x12, 0xaf,
	0x01, 0x0a, 0x03, 0x47, 0x65, 0x74, 0x12, 0x23, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72,
	0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x65, 0x74, 0x45, 0x6e, 0x64, 0x44, 0x65,
	0x76, 0x69, 0x63, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x19, 0x2e, 0x74, 0x74,
	0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64,
	0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x22, 0x68, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x62, 0x12, 0x60,
	0x2f, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x7b, 0x65,
	0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70,
	0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70,
	0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x64, 0x65,
	0x76, 0x69, 0x63, 0x65, 0x73, 0x2f, 0x7b, 0x65, 0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x69, 0x64, 0x7d,
	0x12, 0x74, 0x0a, 0x15, 0x47, 0x65, 0x74, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x66, 0x69, 0x65,
	0x72, 0x73, 0x46, 0x6f, 0x72, 0x45, 0x55, 0x49, 0x73, 0x12, 0x35, 0x2e, 0x74, 0x74, 0x6e, 0x2e,
	0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x47, 0x65, 0x74, 0x45, 0x6e,
	0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x66, 0x69, 0x65,
	0x72, 0x73, 0x46, 0x6f, 0x72, 0x45, 0x55, 0x49, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74,
	0x1a, 0x24, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76,
	0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x49, 0x64, 0x65, 0x6e, 0x74,
	0x69, 0x66, 0x69, 0x65, 0x72, 0x73, 0x12, 0x89, 0x01, 0x0a, 0x04, 0x4c, 0x69, 0x73, 0x74, 0x12,
	0x25, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33,
	0x2e, 0x4c, 0x69, 0x73, 0x74, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1a, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72,
	0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x73, 0x22, 0x3e, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x38, 0x12, 0x36, 0x2f, 0x61, 0x70, 0x70,
	0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x7b, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x73, 0x12, 0xb8, 0x01, 0x0a, 0x06, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65, 0x12, 0x26, 0x2e,
	0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x55,
	0x70, 0x64, 0x61, 0x74, 0x65, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x19, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61,
	0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65,
	0x22, 0x6b, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x65, 0x3a, 0x01, 0x2a, 0x1a, 0x60, 0x2f, 0x61, 0x70,
	0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x7b, 0x65, 0x6e, 0x64, 0x5f,
	0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x73, 0x2f, 0x7b, 0x65, 0x6e, 0x64, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x69,
	0x64, 0x73, 0x2e, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x69, 0x64, 0x7d, 0x12, 0x62, 0x0a,
	0x13, 0x42, 0x61, 0x74, 0x63, 0x68, 0x55, 0x70, 0x64, 0x61, 0x74, 0x65, 0x4c, 0x61, 0x73, 0x74,
	0x53, 0x65, 0x65, 0x6e, 0x12, 0x33, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77,
	0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x42, 0x61, 0x74, 0x63, 0x68, 0x55, 0x70, 0x64, 0x61, 0x74,
	0x65, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x4c, 0x61, 0x73, 0x74, 0x53, 0x65,
	0x65, 0x6e, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67,
	0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74,
	0x79, 0x12, 0x92, 0x01, 0x0a, 0x06, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x12, 0x24, 0x2e, 0x74,
	0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e,
	0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x66, 0x69, 0x65,
	0x72, 0x73, 0x1a, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x22, 0x4a, 0x82, 0xd3, 0xe4, 0x93,
	0x02, 0x44, 0x2a, 0x42, 0x2f, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e,
	0x73, 0x2f, 0x7b, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69,
	0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69,
	0x64, 0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x2f, 0x7b, 0x64, 0x65, 0x76, 0x69,
	0x63, 0x65, 0x5f, 0x69, 0x64, 0x7d, 0x32, 0xff, 0x01, 0x0a, 0x1a, 0x45, 0x6e, 0x64, 0x44, 0x65,
	0x76, 0x69, 0x63, 0x65, 0x54, 0x65, 0x6d, 0x70, 0x6c, 0x61, 0x74, 0x65, 0x43, 0x6f, 0x6e, 0x76,
	0x65, 0x72, 0x74, 0x65, 0x72, 0x12, 0x66, 0x0a, 0x0b, 0x4c, 0x69, 0x73, 0x74, 0x46, 0x6f, 0x72,
	0x6d, 0x61, 0x74, 0x73, 0x12, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x1a, 0x28, 0x2e, 0x74,
	0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e,
	0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x54, 0x65, 0x6d, 0x70, 0x6c, 0x61, 0x74, 0x65, 0x46,
	0x6f, 0x72, 0x6d, 0x61, 0x74, 0x73, 0x22, 0x15, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x0f, 0x12, 0x0d,
	0x2f, 0x65, 0x64, 0x74, 0x63, 0x2f, 0x66, 0x6f, 0x72, 0x6d, 0x61, 0x74, 0x73, 0x12, 0x79, 0x0a,
	0x07, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x12, 0x2f, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c,
	0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72,
	0x74, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x54, 0x65, 0x6d, 0x70, 0x6c, 0x61,
	0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x74, 0x74, 0x6e, 0x2e,
	0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65,
	0x76, 0x69, 0x63, 0x65, 0x54, 0x65, 0x6d, 0x70, 0x6c, 0x61, 0x74, 0x65, 0x22, 0x18, 0x82, 0xd3,
	0xe4, 0x93, 0x02, 0x12, 0x3a, 0x01, 0x2a, 0x22, 0x0d, 0x2f, 0x65, 0x64, 0x74, 0x63, 0x2f, 0x63,
	0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x30, 0x01, 0x32, 0xc4, 0x02, 0x0a, 0x16, 0x45, 0x6e, 0x64,
	0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x42, 0x61, 0x74, 0x63, 0x68, 0x52, 0x65, 0x67, 0x69, 0x73,
	0x74, 0x72, 0x79, 0x12, 0x92, 0x01, 0x0a, 0x03, 0x47, 0x65, 0x74, 0x12, 0x29, 0x2e, 0x74, 0x74,
	0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x42, 0x61, 0x74,
	0x63, 0x68, 0x47, 0x65, 0x74, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1a, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72,
	0x61, 0x77, 0x61, 0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x73, 0x22, 0x44, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x3e, 0x12, 0x3c, 0x2f, 0x61, 0x70, 0x70,
	0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2f, 0x7b, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69,
	0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x73, 0x2f, 0x62, 0x61, 0x74, 0x63, 0x68, 0x12, 0x94, 0x01, 0x0a, 0x06, 0x44, 0x65, 0x6c,
	0x65, 0x74, 0x65, 0x12, 0x2c, 0x2e, 0x74, 0x74, 0x6e, 0x2e, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61,
	0x6e, 0x2e, 0x76, 0x33, 0x2e, 0x42, 0x61, 0x74, 0x63, 0x68, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65,
	0x45, 0x6e, 0x64, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x1a, 0x16, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x75, 0x66, 0x2e, 0x45, 0x6d, 0x70, 0x74, 0x79, 0x22, 0x44, 0x82, 0xd3, 0xe4, 0x93, 0x02,
	0x3e, 0x2a, 0x3c, 0x2f, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73,
	0x2f, 0x7b, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64,
	0x73, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x64,
	0x7d, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x2f, 0x62, 0x61, 0x74, 0x63, 0x68, 0x42,
	0x31, 0x5a, 0x2f, 0x67, 0x6f, 0x2e, 0x74, 0x68, 0x65, 0x74, 0x68, 0x69, 0x6e, 0x67, 0x73, 0x2e,
	0x6e, 0x65, 0x74, 0x77, 0x6f, 0x72, 0x6b, 0x2f, 0x6c, 0x6f, 0x72, 0x61, 0x77, 0x61, 0x6e, 0x2d,
	0x73, 0x74, 0x61, 0x63, 0x6b, 0x2f, 0x76, 0x33, 0x2f, 0x70, 0x6b, 0x67, 0x2f, 0x74, 0x74, 0x6e,
	0x70, 0x62, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var file_ttn_lorawan_v3_end_device_services_proto_goTypes = []interface{}{
	(*CreateEndDeviceRequest)(nil),                // 0: ttn.lorawan.v3.CreateEndDeviceRequest
	(*GetEndDeviceRequest)(nil),                   // 1: ttn.lorawan.v3.GetEndDeviceRequest
	(*GetEndDeviceIdentifiersForEUIsRequest)(nil), // 2: ttn.lorawan.v3.GetEndDeviceIdentifiersForEUIsRequest
	(*ListEndDevicesRequest)(nil),                 // 3: ttn.lorawan.v3.ListEndDevicesRequest
	(*UpdateEndDeviceRequest)(nil),                // 4: ttn.lorawan.v3.UpdateEndDeviceRequest
	(*BatchUpdateEndDeviceLastSeenRequest)(nil),   // 5: ttn.lorawan.v3.BatchUpdateEndDeviceLastSeenRequest
	(*EndDeviceIdentifiers)(nil),                  // 6: ttn.lorawan.v3.EndDeviceIdentifiers
	(*emptypb.Empty)(nil),                         // 7: google.protobuf.Empty
	(*ConvertEndDeviceTemplateRequest)(nil),       // 8: ttn.lorawan.v3.ConvertEndDeviceTemplateRequest
	(*BatchGetEndDevicesRequest)(nil),             // 9: ttn.lorawan.v3.BatchGetEndDevicesRequest
	(*BatchDeleteEndDevicesRequest)(nil),          // 10: ttn.lorawan.v3.BatchDeleteEndDevicesRequest
	(*EndDevice)(nil),                             // 11: ttn.lorawan.v3.EndDevice
	(*EndDevices)(nil),                            // 12: ttn.lorawan.v3.EndDevices
	(*EndDeviceTemplateFormats)(nil),              // 13: ttn.lorawan.v3.EndDeviceTemplateFormats
	(*EndDeviceTemplate)(nil),                     // 14: ttn.lorawan.v3.EndDeviceTemplate
}
var file_ttn_lorawan_v3_end_device_services_proto_depIdxs = []int32{
	0,  // 0: ttn.lorawan.v3.EndDeviceRegistry.Create:input_type -> ttn.lorawan.v3.CreateEndDeviceRequest
	1,  // 1: ttn.lorawan.v3.EndDeviceRegistry.Get:input_type -> ttn.lorawan.v3.GetEndDeviceRequest
	2,  // 2: ttn.lorawan.v3.EndDeviceRegistry.GetIdentifiersForEUIs:input_type -> ttn.lorawan.v3.GetEndDeviceIdentifiersForEUIsRequest
	3,  // 3: ttn.lorawan.v3.EndDeviceRegistry.List:input_type -> ttn.lorawan.v3.ListEndDevicesRequest
	4,  // 4: ttn.lorawan.v3.EndDeviceRegistry.Update:input_type -> ttn.lorawan.v3.UpdateEndDeviceRequest
	5,  // 5: ttn.lorawan.v3.EndDeviceRegistry.BatchUpdateLastSeen:input_type -> ttn.lorawan.v3.BatchUpdateEndDeviceLastSeenRequest
	6,  // 6: ttn.lorawan.v3.EndDeviceRegistry.Delete:input_type -> ttn.lorawan.v3.EndDeviceIdentifiers
	7,  // 7: ttn.lorawan.v3.EndDeviceTemplateConverter.ListFormats:input_type -> google.protobuf.Empty
	8,  // 8: ttn.lorawan.v3.EndDeviceTemplateConverter.Convert:input_type -> ttn.lorawan.v3.ConvertEndDeviceTemplateRequest
	9,  // 9: ttn.lorawan.v3.EndDeviceBatchRegistry.Get:input_type -> ttn.lorawan.v3.BatchGetEndDevicesRequest
	10, // 10: ttn.lorawan.v3.EndDeviceBatchRegistry.Delete:input_type -> ttn.lorawan.v3.BatchDeleteEndDevicesRequest
	11, // 11: ttn.lorawan.v3.EndDeviceRegistry.Create:output_type -> ttn.lorawan.v3.EndDevice
	11, // 12: ttn.lorawan.v3.EndDeviceRegistry.Get:output_type -> ttn.lorawan.v3.EndDevice
	6,  // 13: ttn.lorawan.v3.EndDeviceRegistry.GetIdentifiersForEUIs:output_type -> ttn.lorawan.v3.EndDeviceIdentifiers
	12, // 14: ttn.lorawan.v3.EndDeviceRegistry.List:output_type -> ttn.lorawan.v3.EndDevices
	11, // 15: ttn.lorawan.v3.EndDeviceRegistry.Update:output_type -> ttn.lorawan.v3.EndDevice
	7,  // 16: ttn.lorawan.v3.EndDeviceRegistry.BatchUpdateLastSeen:output_type -> google.protobuf.Empty
	7,  // 17: ttn.lorawan.v3.EndDeviceRegistry.Delete:output_type -> google.protobuf.Empty
	13, // 18: ttn.lorawan.v3.EndDeviceTemplateConverter.ListFormats:output_type -> ttn.lorawan.v3.EndDeviceTemplateFormats
	14, // 19: ttn.lorawan.v3.EndDeviceTemplateConverter.Convert:output_type -> ttn.lorawan.v3.EndDeviceTemplate
	12, // 20: ttn.lorawan.v3.EndDeviceBatchRegistry.Get:output_type -> ttn.lorawan.v3.EndDevices
	7,  // 21: ttn.lorawan.v3.EndDeviceBatchRegistry.Delete:output_type -> google.protobuf.Empty
	11, // [11:22] is the sub-list for method output_type
	0,  // [0:11] is the sub-list for method input_type
	0,  // [0:0] is the sub-list for extension type_name
	0,  // [0:0] is the sub-list for extension extendee
	0,  // [0:0] is the sub-list for field type_name
}

func init() { file_ttn_lorawan_v3_end_device_services_proto_init() }
func file_ttn_lorawan_v3_end_device_services_proto_init() {
	if File_ttn_lorawan_v3_end_device_services_proto != nil {
		return
	}
	file_ttn_lorawan_v3_end_device_proto_init()
	file_ttn_lorawan_v3_identifiers_proto_init()
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_ttn_lorawan_v3_end_device_services_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   0,
			NumExtensions: 0,
			NumServices:   3,
		},
		GoTypes:           file_ttn_lorawan_v3_end_device_services_proto_goTypes,
		DependencyIndexes: file_ttn_lorawan_v3_end_device_services_proto_depIdxs,
	}.Build()
	File_ttn_lorawan_v3_end_device_services_proto = out.File
	file_ttn_lorawan_v3_end_device_services_proto_rawDesc = nil
	file_ttn_lorawan_v3_end_device_services_proto_goTypes = nil
	file_ttn_lorawan_v3_end_device_services_proto_depIdxs = nil
}
