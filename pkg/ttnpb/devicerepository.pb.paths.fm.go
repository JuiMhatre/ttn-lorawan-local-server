// Code generated by protoc-gen-fieldmask. DO NOT EDIT.

package ttnpb

var EndDeviceBrandFieldPathsNested = []string{
	"brand_id",
	"email",
	"logo",
	"lora_alliance_vendor_id",
	"name",
	"organization_unique_identifiers",
	"private_enterprise_number",
	"website",
}

var EndDeviceBrandFieldPathsTopLevel = []string{
	"brand_id",
	"email",
	"logo",
	"lora_alliance_vendor_id",
	"name",
	"organization_unique_identifiers",
	"private_enterprise_number",
	"website",
}
var EndDeviceModelFieldPathsNested = []string{
	"additional_radios",
	"battery",
	"battery.replaceable",
	"battery.type",
	"brand_id",
	"compliances",
	"compliances.radio_equipment",
	"compliances.safety",
	"datasheet_url",
	"description",
	"dimensions",
	"dimensions.diameter",
	"dimensions.height",
	"dimensions.length",
	"dimensions.width",
	"firmware_versions",
	"hardware_versions",
	"ip_code",
	"key_provisioning",
	"key_security",
	"model_id",
	"name",
	"operating_conditions",
	"operating_conditions.relative_humidity",
	"operating_conditions.relative_humidity.max",
	"operating_conditions.relative_humidity.min",
	"operating_conditions.temperature",
	"operating_conditions.temperature.max",
	"operating_conditions.temperature.min",
	"photos",
	"photos.main",
	"photos.other",
	"product_url",
	"resellers",
	"sensors",
	"videos",
	"videos.main",
	"videos.other",
	"weight",
}

var EndDeviceModelFieldPathsTopLevel = []string{
	"additional_radios",
	"battery",
	"brand_id",
	"compliances",
	"datasheet_url",
	"description",
	"dimensions",
	"firmware_versions",
	"hardware_versions",
	"ip_code",
	"key_provisioning",
	"key_security",
	"model_id",
	"name",
	"operating_conditions",
	"photos",
	"product_url",
	"resellers",
	"sensors",
	"videos",
	"weight",
}
var GetEndDeviceBrandRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"brand_id",
	"field_mask",
}

var GetEndDeviceBrandRequestFieldPathsTopLevel = []string{
	"application_ids",
	"brand_id",
	"field_mask",
}
var ListEndDeviceBrandsRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"field_mask",
	"limit",
	"order_by",
	"page",
	"search",
}

var ListEndDeviceBrandsRequestFieldPathsTopLevel = []string{
	"application_ids",
	"field_mask",
	"limit",
	"order_by",
	"page",
	"search",
}
var GetEndDeviceModelRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"brand_id",
	"field_mask",
	"model_id",
}

var GetEndDeviceModelRequestFieldPathsTopLevel = []string{
	"application_ids",
	"brand_id",
	"field_mask",
	"model_id",
}
var ListEndDeviceModelsRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"brand_id",
	"field_mask",
	"limit",
	"order_by",
	"page",
	"search",
}

var ListEndDeviceModelsRequestFieldPathsTopLevel = []string{
	"application_ids",
	"brand_id",
	"field_mask",
	"limit",
	"order_by",
	"page",
	"search",
}
var GetTemplateRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"version_ids",
	"version_ids.band_id",
	"version_ids.brand_id",
	"version_ids.firmware_version",
	"version_ids.hardware_version",
	"version_ids.model_id",
}

var GetTemplateRequestFieldPathsTopLevel = []string{
	"application_ids",
	"version_ids",
}
var GetPayloadFormatterRequestFieldPathsNested = []string{
	"application_ids",
	"application_ids.application_id",
	"field_mask",
	"version_ids",
	"version_ids.band_id",
	"version_ids.brand_id",
	"version_ids.firmware_version",
	"version_ids.hardware_version",
	"version_ids.model_id",
}

var GetPayloadFormatterRequestFieldPathsTopLevel = []string{
	"application_ids",
	"field_mask",
	"version_ids",
}
var ListEndDeviceBrandsResponseFieldPathsNested = []string{
	"brands",
}

var ListEndDeviceBrandsResponseFieldPathsTopLevel = []string{
	"brands",
}
var ListEndDeviceModelsResponseFieldPathsNested = []string{
	"models",
}

var ListEndDeviceModelsResponseFieldPathsTopLevel = []string{
	"models",
}
var EncodedMessagePayloadFieldPathsNested = []string{
	"errors",
	"f_port",
	"frm_payload",
	"warnings",
}

var EncodedMessagePayloadFieldPathsTopLevel = []string{
	"errors",
	"f_port",
	"frm_payload",
	"warnings",
}
var DecodedMessagePayloadFieldPathsNested = []string{
	"data",
	"errors",
	"warnings",
}

var DecodedMessagePayloadFieldPathsTopLevel = []string{
	"data",
	"errors",
	"warnings",
}
var MessagePayloadDecoderFieldPathsNested = []string{
	"codec_id",
	"examples",
	"formatter",
	"formatter_parameter",
}

var MessagePayloadDecoderFieldPathsTopLevel = []string{
	"codec_id",
	"examples",
	"formatter",
	"formatter_parameter",
}
var MessagePayloadEncoderFieldPathsNested = []string{
	"codec_id",
	"examples",
	"formatter",
	"formatter_parameter",
}

var MessagePayloadEncoderFieldPathsTopLevel = []string{
	"codec_id",
	"examples",
	"formatter",
	"formatter_parameter",
}
var EndDeviceModel_HardwareVersionFieldPathsNested = []string{
	"numeric",
	"part_number",
	"version",
}

var EndDeviceModel_HardwareVersionFieldPathsTopLevel = []string{
	"numeric",
	"part_number",
	"version",
}
var EndDeviceModel_FirmwareVersionFieldPathsNested = []string{
	"numeric",
	"profiles",
	"supported_hardware_versions",
	"version",
}

var EndDeviceModel_FirmwareVersionFieldPathsTopLevel = []string{
	"numeric",
	"profiles",
	"supported_hardware_versions",
	"version",
}
var EndDeviceModel_DimensionsFieldPathsNested = []string{
	"diameter",
	"height",
	"length",
	"width",
}

var EndDeviceModel_DimensionsFieldPathsTopLevel = []string{
	"diameter",
	"height",
	"length",
	"width",
}
var EndDeviceModel_BatteryFieldPathsNested = []string{
	"replaceable",
	"type",
}

var EndDeviceModel_BatteryFieldPathsTopLevel = []string{
	"replaceable",
	"type",
}
var EndDeviceModel_OperatingConditionsFieldPathsNested = []string{
	"relative_humidity",
	"relative_humidity.max",
	"relative_humidity.min",
	"temperature",
	"temperature.max",
	"temperature.min",
}

var EndDeviceModel_OperatingConditionsFieldPathsTopLevel = []string{
	"relative_humidity",
	"temperature",
}
var EndDeviceModel_PhotosFieldPathsNested = []string{
	"main",
	"other",
}

var EndDeviceModel_PhotosFieldPathsTopLevel = []string{
	"main",
	"other",
}
var EndDeviceModel_VideosFieldPathsNested = []string{
	"main",
	"other",
}

var EndDeviceModel_VideosFieldPathsTopLevel = []string{
	"main",
	"other",
}
var EndDeviceModel_ResellerFieldPathsNested = []string{
	"name",
	"region",
	"url",
}

var EndDeviceModel_ResellerFieldPathsTopLevel = []string{
	"name",
	"region",
	"url",
}
var EndDeviceModel_CompliancesFieldPathsNested = []string{
	"radio_equipment",
	"safety",
}

var EndDeviceModel_CompliancesFieldPathsTopLevel = []string{
	"radio_equipment",
	"safety",
}
var EndDeviceModel_FirmwareVersion_ProfileFieldPathsNested = []string{
	"codec_id",
	"lorawan_certified",
	"profile_id",
	"vendor_id",
}

var EndDeviceModel_FirmwareVersion_ProfileFieldPathsTopLevel = []string{
	"codec_id",
	"lorawan_certified",
	"profile_id",
	"vendor_id",
}
var EndDeviceModel_OperatingConditions_LimitsFieldPathsNested = []string{
	"max",
	"min",
}

var EndDeviceModel_OperatingConditions_LimitsFieldPathsTopLevel = []string{
	"max",
	"min",
}
var EndDeviceModel_Compliances_ComplianceFieldPathsNested = []string{
	"body",
	"norm",
	"standard",
	"version",
}

var EndDeviceModel_Compliances_ComplianceFieldPathsTopLevel = []string{
	"body",
	"norm",
	"standard",
	"version",
}
var MessagePayloadDecoder_ExampleFieldPathsNested = []string{
	"description",
	"input",
	"input.errors",
	"input.f_port",
	"input.frm_payload",
	"input.warnings",
	"output",
	"output.data",
	"output.errors",
	"output.warnings",
}

var MessagePayloadDecoder_ExampleFieldPathsTopLevel = []string{
	"description",
	"input",
	"output",
}
var MessagePayloadEncoder_ExampleFieldPathsNested = []string{
	"description",
	"input",
	"input.data",
	"input.errors",
	"input.warnings",
	"output",
	"output.errors",
	"output.f_port",
	"output.frm_payload",
	"output.warnings",
}

var MessagePayloadEncoder_ExampleFieldPathsTopLevel = []string{
	"description",
	"input",
	"output",
}
