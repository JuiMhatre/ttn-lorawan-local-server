// Code generated by protoc-gen-fieldmask. DO NOT EDIT.

package simulate

var SimulateMetadataParamsFieldPathsNested = []string{
	"band_id",
	"bandwidth",
	"channel_index",
	"data_rate_index",
	"frequency",
	"lorawan_phy_version",
	"lorawan_version",
	"rssi",
	"snr",
	"spreading_factor",
	"time",
	"timestamp",
}

var SimulateMetadataParamsFieldPathsTopLevel = []string{
	"band_id",
	"bandwidth",
	"channel_index",
	"data_rate_index",
	"frequency",
	"lorawan_phy_version",
	"lorawan_version",
	"rssi",
	"snr",
	"spreading_factor",
	"time",
	"timestamp",
}
var SimulateJoinRequestParamsFieldPathsNested = []string{
	"app_key",
	"app_key.encrypted_key",
	"app_key.kek_label",
	"app_key.key",
	"dev_eui",
	"dev_nonce",
	"join_eui",
	"nwk_key",
	"nwk_key.encrypted_key",
	"nwk_key.kek_label",
	"nwk_key.key",
}

var SimulateJoinRequestParamsFieldPathsTopLevel = []string{
	"app_key",
	"dev_eui",
	"dev_nonce",
	"join_eui",
	"nwk_key",
}
var SimulateDataUplinkParamsFieldPathsNested = []string{
	"ack",
	"adr",
	"adr_ack_req",
	"app_s_key",
	"app_s_key.encrypted_key",
	"app_s_key.kek_label",
	"app_s_key.key",
	"conf_f_cnt",
	"confirmed",
	"dev_addr",
	"f_cnt",
	"f_nwk_s_int_key",
	"f_nwk_s_int_key.encrypted_key",
	"f_nwk_s_int_key.kek_label",
	"f_nwk_s_int_key.key",
	"f_opts",
	"f_port",
	"frm_payload",
	"nwk_s_enc_key",
	"nwk_s_enc_key.encrypted_key",
	"nwk_s_enc_key.kek_label",
	"nwk_s_enc_key.key",
	"s_nwk_s_int_key",
	"s_nwk_s_int_key.encrypted_key",
	"s_nwk_s_int_key.kek_label",
	"s_nwk_s_int_key.key",
	"tx_ch_idx",
	"tx_dr_idx",
}

var SimulateDataUplinkParamsFieldPathsTopLevel = []string{
	"ack",
	"adr",
	"adr_ack_req",
	"app_s_key",
	"conf_f_cnt",
	"confirmed",
	"dev_addr",
	"f_cnt",
	"f_nwk_s_int_key",
	"f_opts",
	"f_port",
	"frm_payload",
	"nwk_s_enc_key",
	"s_nwk_s_int_key",
	"tx_ch_idx",
	"tx_dr_idx",
}
