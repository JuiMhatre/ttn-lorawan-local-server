// Code generated by protoc-gen-fieldmask. DO NOT EDIT.

package ttnpb

var GatewayUpFieldPathsNested = []string{
	"gateway_status",
	"gateway_status.advanced",
	"gateway_status.antenna_locations",
	"gateway_status.boot_time",
	"gateway_status.ip",
	"gateway_status.metrics",
	"gateway_status.time",
	"gateway_status.versions",
	"tx_acknowledgment",
	"tx_acknowledgment.correlation_ids",
	"tx_acknowledgment.downlink_message",
	"tx_acknowledgment.downlink_message.correlation_ids",
	"tx_acknowledgment.downlink_message.end_device_ids",
	"tx_acknowledgment.downlink_message.end_device_ids.application_ids",
	"tx_acknowledgment.downlink_message.end_device_ids.application_ids.application_id",
	"tx_acknowledgment.downlink_message.end_device_ids.dev_addr",
	"tx_acknowledgment.downlink_message.end_device_ids.dev_eui",
	"tx_acknowledgment.downlink_message.end_device_ids.device_id",
	"tx_acknowledgment.downlink_message.end_device_ids.join_eui",
	"tx_acknowledgment.downlink_message.payload",
	"tx_acknowledgment.downlink_message.payload.Payload",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.cf_list",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.cf_list.ch_masks",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.cf_list.freq",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.cf_list.type",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.dev_addr",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.dl_settings",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.dl_settings.opt_neg",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.dl_settings.rx1_dr_offset",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.dl_settings.rx2_dr",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.encrypted",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.join_nonce",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.net_id",
	"tx_acknowledgment.downlink_message.payload.Payload.join_accept_payload.rx_delay",
	"tx_acknowledgment.downlink_message.payload.Payload.join_request_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.join_request_payload.dev_eui",
	"tx_acknowledgment.downlink_message.payload.Payload.join_request_payload.dev_nonce",
	"tx_acknowledgment.downlink_message.payload.Payload.join_request_payload.join_eui",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.decoded_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.dev_addr",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_cnt",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.ack",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.adr",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.adr_ack_req",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.class_b",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.f_pending",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_hdr.f_opts",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.f_port",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.frm_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.mac_payload.full_f_cnt",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload.dev_eui",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload.join_eui",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload.net_id",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload.rejoin_cnt",
	"tx_acknowledgment.downlink_message.payload.Payload.rejoin_request_payload.rejoin_type",
	"tx_acknowledgment.downlink_message.payload.m_hdr",
	"tx_acknowledgment.downlink_message.payload.m_hdr.m_type",
	"tx_acknowledgment.downlink_message.payload.m_hdr.major",
	"tx_acknowledgment.downlink_message.payload.mic",
	"tx_acknowledgment.downlink_message.raw_payload",
	"tx_acknowledgment.downlink_message.session_key_id",
	"tx_acknowledgment.downlink_message.settings",
	"tx_acknowledgment.downlink_message.settings.request",
	"tx_acknowledgment.downlink_message.settings.request.absolute_time",
	"tx_acknowledgment.downlink_message.settings.request.advanced",
	"tx_acknowledgment.downlink_message.settings.request.class",
	"tx_acknowledgment.downlink_message.settings.request.downlink_paths",
	"tx_acknowledgment.downlink_message.settings.request.frequency_plan_id",
	"tx_acknowledgment.downlink_message.settings.request.priority",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.fsk",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.fsk.bit_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lora",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lora.bandwidth",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lora.coding_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lora.spreading_factor",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lrfhss",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.coding_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.modulation_type",
	"tx_acknowledgment.downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.operating_channel_width",
	"tx_acknowledgment.downlink_message.settings.request.rx1_delay",
	"tx_acknowledgment.downlink_message.settings.request.rx1_frequency",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.fsk",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.fsk.bit_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lora",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lora.bandwidth",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lora.coding_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lora.spreading_factor",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lrfhss",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.coding_rate",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.modulation_type",
	"tx_acknowledgment.downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.operating_channel_width",
	"tx_acknowledgment.downlink_message.settings.request.rx2_frequency",
	"tx_acknowledgment.downlink_message.settings.scheduled",
	"tx_acknowledgment.downlink_message.settings.scheduled.concentrator_timestamp",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.fsk",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.fsk.bit_rate",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lora",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lora.bandwidth",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lora.coding_rate",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lora.spreading_factor",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lrfhss",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lrfhss.coding_rate",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lrfhss.modulation_type",
	"tx_acknowledgment.downlink_message.settings.scheduled.data_rate.modulation.lrfhss.operating_channel_width",
	"tx_acknowledgment.downlink_message.settings.scheduled.downlink",
	"tx_acknowledgment.downlink_message.settings.scheduled.downlink.antenna_index",
	"tx_acknowledgment.downlink_message.settings.scheduled.downlink.invert_polarization",
	"tx_acknowledgment.downlink_message.settings.scheduled.downlink.tx_power",
	"tx_acknowledgment.downlink_message.settings.scheduled.enable_crc",
	"tx_acknowledgment.downlink_message.settings.scheduled.frequency",
	"tx_acknowledgment.downlink_message.settings.scheduled.time",
	"tx_acknowledgment.downlink_message.settings.scheduled.timestamp",
	"tx_acknowledgment.result",
	"uplink_messages",
}

var GatewayUpFieldPathsTopLevel = []string{
	"gateway_status",
	"tx_acknowledgment",
	"uplink_messages",
}
var GatewayDownFieldPathsNested = []string{
	"downlink_message",
	"downlink_message.correlation_ids",
	"downlink_message.end_device_ids",
	"downlink_message.end_device_ids.application_ids",
	"downlink_message.end_device_ids.application_ids.application_id",
	"downlink_message.end_device_ids.dev_addr",
	"downlink_message.end_device_ids.dev_eui",
	"downlink_message.end_device_ids.device_id",
	"downlink_message.end_device_ids.join_eui",
	"downlink_message.payload",
	"downlink_message.payload.Payload",
	"downlink_message.payload.Payload.join_accept_payload",
	"downlink_message.payload.Payload.join_accept_payload.cf_list",
	"downlink_message.payload.Payload.join_accept_payload.cf_list.ch_masks",
	"downlink_message.payload.Payload.join_accept_payload.cf_list.freq",
	"downlink_message.payload.Payload.join_accept_payload.cf_list.type",
	"downlink_message.payload.Payload.join_accept_payload.dev_addr",
	"downlink_message.payload.Payload.join_accept_payload.dl_settings",
	"downlink_message.payload.Payload.join_accept_payload.dl_settings.opt_neg",
	"downlink_message.payload.Payload.join_accept_payload.dl_settings.rx1_dr_offset",
	"downlink_message.payload.Payload.join_accept_payload.dl_settings.rx2_dr",
	"downlink_message.payload.Payload.join_accept_payload.encrypted",
	"downlink_message.payload.Payload.join_accept_payload.join_nonce",
	"downlink_message.payload.Payload.join_accept_payload.net_id",
	"downlink_message.payload.Payload.join_accept_payload.rx_delay",
	"downlink_message.payload.Payload.join_request_payload",
	"downlink_message.payload.Payload.join_request_payload.dev_eui",
	"downlink_message.payload.Payload.join_request_payload.dev_nonce",
	"downlink_message.payload.Payload.join_request_payload.join_eui",
	"downlink_message.payload.Payload.mac_payload",
	"downlink_message.payload.Payload.mac_payload.decoded_payload",
	"downlink_message.payload.Payload.mac_payload.f_hdr",
	"downlink_message.payload.Payload.mac_payload.f_hdr.dev_addr",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_cnt",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.ack",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.adr",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.adr_ack_req",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.class_b",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_ctrl.f_pending",
	"downlink_message.payload.Payload.mac_payload.f_hdr.f_opts",
	"downlink_message.payload.Payload.mac_payload.f_port",
	"downlink_message.payload.Payload.mac_payload.frm_payload",
	"downlink_message.payload.Payload.mac_payload.full_f_cnt",
	"downlink_message.payload.Payload.rejoin_request_payload",
	"downlink_message.payload.Payload.rejoin_request_payload.dev_eui",
	"downlink_message.payload.Payload.rejoin_request_payload.join_eui",
	"downlink_message.payload.Payload.rejoin_request_payload.net_id",
	"downlink_message.payload.Payload.rejoin_request_payload.rejoin_cnt",
	"downlink_message.payload.Payload.rejoin_request_payload.rejoin_type",
	"downlink_message.payload.m_hdr",
	"downlink_message.payload.m_hdr.m_type",
	"downlink_message.payload.m_hdr.major",
	"downlink_message.payload.mic",
	"downlink_message.raw_payload",
	"downlink_message.session_key_id",
	"downlink_message.settings",
	"downlink_message.settings.request",
	"downlink_message.settings.request.absolute_time",
	"downlink_message.settings.request.advanced",
	"downlink_message.settings.request.class",
	"downlink_message.settings.request.downlink_paths",
	"downlink_message.settings.request.frequency_plan_id",
	"downlink_message.settings.request.priority",
	"downlink_message.settings.request.rx1_data_rate",
	"downlink_message.settings.request.rx1_data_rate.modulation",
	"downlink_message.settings.request.rx1_data_rate.modulation.fsk",
	"downlink_message.settings.request.rx1_data_rate.modulation.fsk.bit_rate",
	"downlink_message.settings.request.rx1_data_rate.modulation.lora",
	"downlink_message.settings.request.rx1_data_rate.modulation.lora.bandwidth",
	"downlink_message.settings.request.rx1_data_rate.modulation.lora.coding_rate",
	"downlink_message.settings.request.rx1_data_rate.modulation.lora.spreading_factor",
	"downlink_message.settings.request.rx1_data_rate.modulation.lrfhss",
	"downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.coding_rate",
	"downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.modulation_type",
	"downlink_message.settings.request.rx1_data_rate.modulation.lrfhss.operating_channel_width",
	"downlink_message.settings.request.rx1_delay",
	"downlink_message.settings.request.rx1_frequency",
	"downlink_message.settings.request.rx2_data_rate",
	"downlink_message.settings.request.rx2_data_rate.modulation",
	"downlink_message.settings.request.rx2_data_rate.modulation.fsk",
	"downlink_message.settings.request.rx2_data_rate.modulation.fsk.bit_rate",
	"downlink_message.settings.request.rx2_data_rate.modulation.lora",
	"downlink_message.settings.request.rx2_data_rate.modulation.lora.bandwidth",
	"downlink_message.settings.request.rx2_data_rate.modulation.lora.coding_rate",
	"downlink_message.settings.request.rx2_data_rate.modulation.lora.spreading_factor",
	"downlink_message.settings.request.rx2_data_rate.modulation.lrfhss",
	"downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.coding_rate",
	"downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.modulation_type",
	"downlink_message.settings.request.rx2_data_rate.modulation.lrfhss.operating_channel_width",
	"downlink_message.settings.request.rx2_frequency",
	"downlink_message.settings.scheduled",
	"downlink_message.settings.scheduled.concentrator_timestamp",
	"downlink_message.settings.scheduled.data_rate",
	"downlink_message.settings.scheduled.data_rate.modulation",
	"downlink_message.settings.scheduled.data_rate.modulation.fsk",
	"downlink_message.settings.scheduled.data_rate.modulation.fsk.bit_rate",
	"downlink_message.settings.scheduled.data_rate.modulation.lora",
	"downlink_message.settings.scheduled.data_rate.modulation.lora.bandwidth",
	"downlink_message.settings.scheduled.data_rate.modulation.lora.coding_rate",
	"downlink_message.settings.scheduled.data_rate.modulation.lora.spreading_factor",
	"downlink_message.settings.scheduled.data_rate.modulation.lrfhss",
	"downlink_message.settings.scheduled.data_rate.modulation.lrfhss.coding_rate",
	"downlink_message.settings.scheduled.data_rate.modulation.lrfhss.modulation_type",
	"downlink_message.settings.scheduled.data_rate.modulation.lrfhss.operating_channel_width",
	"downlink_message.settings.scheduled.downlink",
	"downlink_message.settings.scheduled.downlink.antenna_index",
	"downlink_message.settings.scheduled.downlink.invert_polarization",
	"downlink_message.settings.scheduled.downlink.tx_power",
	"downlink_message.settings.scheduled.enable_crc",
	"downlink_message.settings.scheduled.frequency",
	"downlink_message.settings.scheduled.time",
	"downlink_message.settings.scheduled.timestamp",
}

var GatewayDownFieldPathsTopLevel = []string{
	"downlink_message",
}
var ScheduleDownlinkResponseFieldPathsNested = []string{
	"delay",
	"downlink_path",
	"downlink_path.path",
	"downlink_path.path.fixed",
	"downlink_path.path.fixed.antenna_index",
	"downlink_path.path.fixed.gateway_ids",
	"downlink_path.path.fixed.gateway_ids.eui",
	"downlink_path.path.fixed.gateway_ids.gateway_id",
	"downlink_path.path.uplink_token",
	"rx1",
	"rx2",
}

var ScheduleDownlinkResponseFieldPathsTopLevel = []string{
	"delay",
	"downlink_path",
	"rx1",
	"rx2",
}
var ScheduleDownlinkErrorDetailsFieldPathsNested = []string{
	"path_errors",
}

var ScheduleDownlinkErrorDetailsFieldPathsTopLevel = []string{
	"path_errors",
}
var BatchGetGatewayConnectionStatsRequestFieldPathsNested = []string{
	"field_mask",
	"gateway_ids",
}

var BatchGetGatewayConnectionStatsRequestFieldPathsTopLevel = []string{
	"field_mask",
	"gateway_ids",
}
var BatchGetGatewayConnectionStatsResponseFieldPathsNested = []string{
	"entries",
}

var BatchGetGatewayConnectionStatsResponseFieldPathsTopLevel = []string{
	"entries",
}
