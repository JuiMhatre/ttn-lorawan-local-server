// Code generated by protoc-gen-go-flags. DO NOT EDIT.
// versions:
// - protoc-gen-go-flags v1.0.6
// - protoc              v3.9.1
// source: lorawan-stack/api/simulate.proto

package simulate

import (
	flagsplugin "github.com/TheThingsIndustries/protoc-gen-go-flags/flagsplugin"
	gogo "github.com/TheThingsIndustries/protoc-gen-go-flags/gogo"
	pflag "github.com/spf13/pflag"
	ttnpb "go.thethings.network/lorawan-stack/v3/pkg/ttnpb"
)

// AddSelectFlagsForSimulateMetadataParams adds flags to select fields in SimulateMetadataParams.
func AddSelectFlagsForSimulateMetadataParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("rssi", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("rssi", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("snr", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("snr", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("timestamp", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("timestamp", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("time", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("time", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("lorawan-version", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("lorawan-version", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("lorawan-phy-version", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("lorawan-phy-version", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("band-id", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("band-id", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("frequency", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("frequency", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("channel-index", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("channel-index", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("bandwidth", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("bandwidth", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("spreading-factor", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("spreading-factor", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("data-rate-index", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("data-rate-index", prefix), false), flagsplugin.WithHidden(hidden)))
}

// SelectFromFlags outputs the fieldmask paths forSimulateMetadataParams message from select flags.
func PathsFromSelectFlagsForSimulateMetadataParams(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("rssi", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("rssi", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("snr", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("snr", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("timestamp", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("timestamp", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("time", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("time", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("lorawan_version", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("lorawan_version", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("lorawan_phy_version", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("lorawan_phy_version", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("band_id", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("band_id", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("frequency", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("frequency", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("channel_index", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("channel_index", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("bandwidth", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("bandwidth", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("spreading_factor", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("spreading_factor", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("data_rate_index", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("data_rate_index", prefix))
	}
	return paths, nil
}

// AddSetFlagsForSimulateMetadataParams adds flags to select fields in SimulateMetadataParams.
func AddSetFlagsForSimulateMetadataParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewFloat32Flag(flagsplugin.Prefix("rssi", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewFloat32Flag(flagsplugin.Prefix("snr", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("timestamp", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewTimestampFlag(flagsplugin.Prefix("time", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("lorawan-version", prefix), flagsplugin.EnumValueDesc(ttnpb.MACVersion_value, ttnpb.MACVersion_customvalue), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("lorawan-phy-version", prefix), flagsplugin.EnumValueDesc(ttnpb.PHYVersion_value, ttnpb.PHYVersion_customvalue), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("band-id", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint64Flag(flagsplugin.Prefix("frequency", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("channel-index", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("bandwidth", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("spreading-factor", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("data-rate-index", prefix), "", flagsplugin.WithHidden(hidden)))
}

// SetFromFlags sets the SimulateMetadataParams message from flags.
func (m *SimulateMetadataParams) SetFromFlags(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, changed, err := flagsplugin.GetFloat32(flags, flagsplugin.Prefix("rssi", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Rssi = val
		paths = append(paths, flagsplugin.Prefix("rssi", prefix))
	}
	if val, changed, err := flagsplugin.GetFloat32(flags, flagsplugin.Prefix("snr", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Snr = val
		paths = append(paths, flagsplugin.Prefix("snr", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("timestamp", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Timestamp = val
		paths = append(paths, flagsplugin.Prefix("timestamp", prefix))
	}
	if val, changed, err := flagsplugin.GetTimestamp(flags, flagsplugin.Prefix("time", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Time = gogo.SetTimestamp(val)
		paths = append(paths, flagsplugin.Prefix("time", prefix))
	}
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("lorawan_version", prefix)); err != nil {
		return nil, err
	} else if changed {
		enumValue, err := flagsplugin.SetEnumString(val, ttnpb.MACVersion_value, ttnpb.MACVersion_customvalue)
		if err != nil {
			return nil, err
		}
		m.LorawanVersion = ttnpb.MACVersion(enumValue)
		paths = append(paths, flagsplugin.Prefix("lorawan_version", prefix))
	}
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("lorawan_phy_version", prefix)); err != nil {
		return nil, err
	} else if changed {
		enumValue, err := flagsplugin.SetEnumString(val, ttnpb.PHYVersion_value, ttnpb.PHYVersion_customvalue)
		if err != nil {
			return nil, err
		}
		m.LorawanPhyVersion = ttnpb.PHYVersion(enumValue)
		paths = append(paths, flagsplugin.Prefix("lorawan_phy_version", prefix))
	}
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("band_id", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.BandId = val
		paths = append(paths, flagsplugin.Prefix("band_id", prefix))
	}
	if val, changed, err := flagsplugin.GetUint64(flags, flagsplugin.Prefix("frequency", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Frequency = val
		paths = append(paths, flagsplugin.Prefix("frequency", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("channel_index", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.ChannelIndex = val
		paths = append(paths, flagsplugin.Prefix("channel_index", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("bandwidth", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Bandwidth = val
		paths = append(paths, flagsplugin.Prefix("bandwidth", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("spreading_factor", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.SpreadingFactor = val
		paths = append(paths, flagsplugin.Prefix("spreading_factor", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("data_rate_index", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.DataRateIndex = val
		paths = append(paths, flagsplugin.Prefix("data_rate_index", prefix))
	}
	return paths, nil
}

// AddSelectFlagsForSimulateJoinRequestParams adds flags to select fields in SimulateJoinRequestParams.
func AddSelectFlagsForSimulateJoinRequestParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("join-eui", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("join-eui", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("dev-eui", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("dev-eui", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("dev-nonce", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("dev-nonce", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("app-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("app-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("nwk-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("nwk-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk-key", prefix), hidden)
}

// SelectFromFlags outputs the fieldmask paths forSimulateJoinRequestParams message from select flags.
func PathsFromSelectFlagsForSimulateJoinRequestParams(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("join_eui", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("join_eui", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("dev_eui", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("dev_eui", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("dev_nonce", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("dev_nonce", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("app_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("app_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("nwk_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("nwk_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	return paths, nil
}

// AddSetFlagsForSimulateJoinRequestParams adds flags to select fields in SimulateJoinRequestParams.
func AddSetFlagsForSimulateJoinRequestParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("join-eui", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("dev-eui", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("dev-nonce", prefix), "", flagsplugin.WithHidden(hidden)))
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app-key", prefix), hidden)
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk-key", prefix), hidden)
}

// SetFromFlags sets the SimulateJoinRequestParams message from flags.
func (m *SimulateJoinRequestParams) SetFromFlags(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("join_eui", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.JoinEui = val
		paths = append(paths, flagsplugin.Prefix("join_eui", prefix))
	}
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("dev_eui", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.DevEui = val
		paths = append(paths, flagsplugin.Prefix("dev_eui", prefix))
	}
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("dev_nonce", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.DevNonce = val
		paths = append(paths, flagsplugin.Prefix("dev_nonce", prefix))
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("app_key", prefix)); changed {
		if m.AppKey == nil {
			m.AppKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.AppKey.SetFromFlags(flags, flagsplugin.Prefix("app_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("nwk_key", prefix)); changed {
		if m.NwkKey == nil {
			m.NwkKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.NwkKey.SetFromFlags(flags, flagsplugin.Prefix("nwk_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	return paths, nil
}

// AddSelectFlagsForSimulateDataUplinkParams adds flags to select fields in SimulateDataUplinkParams.
func AddSelectFlagsForSimulateDataUplinkParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("dev-addr", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("dev-addr", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("f-nwk-s-int-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("f-nwk-s-int-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("f-nwk-s-int-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("s-nwk-s-int-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("s-nwk-s-int-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("s-nwk-s-int-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("nwk-s-enc-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("nwk-s-enc-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk-s-enc-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("app-s-key", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("app-s-key", prefix), true), flagsplugin.WithHidden(hidden)))
	ttnpb.AddSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app-s-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("adr", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("adr", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("adr-ack-req", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("adr-ack-req", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("confirmed", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("confirmed", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("ack", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("ack", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("f-cnt", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("f-cnt", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("f-port", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("f-port", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("frm-payload", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("frm-payload", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("conf-f-cnt", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("conf-f-cnt", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("tx-dr-idx", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("tx-dr-idx", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("tx-ch-idx", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("tx-ch-idx", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("f-opts", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("f-opts", prefix), false), flagsplugin.WithHidden(hidden)))
}

// SelectFromFlags outputs the fieldmask paths forSimulateDataUplinkParams message from select flags.
func PathsFromSelectFlagsForSimulateDataUplinkParams(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("dev_addr", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("dev_addr", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("f_nwk_s_int_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("f_nwk_s_int_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("f_nwk_s_int_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("s_nwk_s_int_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("s_nwk_s_int_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("s_nwk_s_int_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("nwk_s_enc_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("nwk_s_enc_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk_s_enc_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("app_s_key", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("app_s_key", prefix))
	}
	if selectPaths, err := ttnpb.PathsFromSelectFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app_s_key", prefix)); err != nil {
		return nil, err
	} else {
		paths = append(paths, selectPaths...)
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("adr", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("adr", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("adr_ack_req", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("adr_ack_req", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("confirmed", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("confirmed", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("ack", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("ack", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("f_cnt", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("f_cnt", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("f_port", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("f_port", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("frm_payload", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("frm_payload", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("conf_f_cnt", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("conf_f_cnt", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("tx_dr_idx", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("tx_dr_idx", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("tx_ch_idx", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("tx_ch_idx", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("f_opts", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("f_opts", prefix))
	}
	return paths, nil
}

// AddSetFlagsForSimulateDataUplinkParams adds flags to select fields in SimulateDataUplinkParams.
func AddSetFlagsForSimulateDataUplinkParams(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("dev-addr", prefix), "", flagsplugin.WithHidden(hidden)))
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("f-nwk-s-int-key", prefix), hidden)
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("s-nwk-s-int-key", prefix), hidden)
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("nwk-s-enc-key", prefix), hidden)
	ttnpb.AddSetFlagsForKeyEnvelope(flags, flagsplugin.Prefix("app-s-key", prefix), hidden)
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("adr", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("adr-ack-req", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("confirmed", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("ack", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("f-cnt", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("f-port", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("frm-payload", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("conf-f-cnt", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("tx-dr-idx", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewUint32Flag(flagsplugin.Prefix("tx-ch-idx", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBytesFlag(flagsplugin.Prefix("f-opts", prefix), "", flagsplugin.WithHidden(hidden)))
}

// SetFromFlags sets the SimulateDataUplinkParams message from flags.
func (m *SimulateDataUplinkParams) SetFromFlags(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("dev_addr", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.DevAddr = val
		paths = append(paths, flagsplugin.Prefix("dev_addr", prefix))
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("f_nwk_s_int_key", prefix)); changed {
		if m.FNwkSIntKey == nil {
			m.FNwkSIntKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.FNwkSIntKey.SetFromFlags(flags, flagsplugin.Prefix("f_nwk_s_int_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("s_nwk_s_int_key", prefix)); changed {
		if m.SNwkSIntKey == nil {
			m.SNwkSIntKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.SNwkSIntKey.SetFromFlags(flags, flagsplugin.Prefix("s_nwk_s_int_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("nwk_s_enc_key", prefix)); changed {
		if m.NwkSEncKey == nil {
			m.NwkSEncKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.NwkSEncKey.SetFromFlags(flags, flagsplugin.Prefix("nwk_s_enc_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	if changed := flagsplugin.IsAnyPrefixSet(flags, flagsplugin.Prefix("app_s_key", prefix)); changed {
		if m.AppSKey == nil {
			m.AppSKey = &ttnpb.KeyEnvelope{}
		}
		if setPaths, err := m.AppSKey.SetFromFlags(flags, flagsplugin.Prefix("app_s_key", prefix)); err != nil {
			return nil, err
		} else {
			paths = append(paths, setPaths...)
		}
	}
	if val, changed, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("adr", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Adr = val
		paths = append(paths, flagsplugin.Prefix("adr", prefix))
	}
	if val, changed, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("adr_ack_req", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.AdrAckReq = val
		paths = append(paths, flagsplugin.Prefix("adr_ack_req", prefix))
	}
	if val, changed, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("confirmed", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Confirmed = val
		paths = append(paths, flagsplugin.Prefix("confirmed", prefix))
	}
	if val, changed, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("ack", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Ack = val
		paths = append(paths, flagsplugin.Prefix("ack", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("f_cnt", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.FCnt = val
		paths = append(paths, flagsplugin.Prefix("f_cnt", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("f_port", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.FPort = val
		paths = append(paths, flagsplugin.Prefix("f_port", prefix))
	}
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("frm_payload", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.FrmPayload = val
		paths = append(paths, flagsplugin.Prefix("frm_payload", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("conf_f_cnt", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.ConfFCnt = val
		paths = append(paths, flagsplugin.Prefix("conf_f_cnt", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("tx_dr_idx", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.TxDrIdx = val
		paths = append(paths, flagsplugin.Prefix("tx_dr_idx", prefix))
	}
	if val, changed, err := flagsplugin.GetUint32(flags, flagsplugin.Prefix("tx_ch_idx", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.TxChIdx = val
		paths = append(paths, flagsplugin.Prefix("tx_ch_idx", prefix))
	}
	if val, changed, err := flagsplugin.GetBytes(flags, flagsplugin.Prefix("f_opts", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.FOpts = val
		paths = append(paths, flagsplugin.Prefix("f_opts", prefix))
	}
	return paths, nil
}
