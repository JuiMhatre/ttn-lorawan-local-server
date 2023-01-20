// Code generated by protoc-gen-fieldmask. DO NOT EDIT.

package simulate

import (
	fmt "fmt"
	ttnpb "go.thethings.network/lorawan-stack/v3/pkg/ttnpb"
)

func (dst *SimulateMetadataParams) SetFields(src *SimulateMetadataParams, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "rssi":
			if len(subs) > 0 {
				return fmt.Errorf("'rssi' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Rssi = src.Rssi
			} else {
				var zero float32
				dst.Rssi = zero
			}
		case "snr":
			if len(subs) > 0 {
				return fmt.Errorf("'snr' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Snr = src.Snr
			} else {
				var zero float32
				dst.Snr = zero
			}
		case "timestamp":
			if len(subs) > 0 {
				return fmt.Errorf("'timestamp' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Timestamp = src.Timestamp
			} else {
				var zero uint32
				dst.Timestamp = zero
			}
		case "time":
			if len(subs) > 0 {
				return fmt.Errorf("'time' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Time = src.Time
			} else {
				dst.Time = nil
			}
		case "lorawan_version":
			if len(subs) > 0 {
				return fmt.Errorf("'lorawan_version' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.LorawanVersion = src.LorawanVersion
			} else {
				var zero ttnpb.MACVersion
				dst.LorawanVersion = zero
			}
		case "lorawan_phy_version":
			if len(subs) > 0 {
				return fmt.Errorf("'lorawan_phy_version' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.LorawanPhyVersion = src.LorawanPhyVersion
			} else {
				var zero ttnpb.PHYVersion
				dst.LorawanPhyVersion = zero
			}
		case "band_id":
			if len(subs) > 0 {
				return fmt.Errorf("'band_id' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.BandId = src.BandId
			} else {
				var zero string
				dst.BandId = zero
			}
		case "frequency":
			if len(subs) > 0 {
				return fmt.Errorf("'frequency' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Frequency = src.Frequency
			} else {
				var zero uint64
				dst.Frequency = zero
			}
		case "channel_index":
			if len(subs) > 0 {
				return fmt.Errorf("'channel_index' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ChannelIndex = src.ChannelIndex
			} else {
				var zero uint32
				dst.ChannelIndex = zero
			}
		case "bandwidth":
			if len(subs) > 0 {
				return fmt.Errorf("'bandwidth' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Bandwidth = src.Bandwidth
			} else {
				var zero uint32
				dst.Bandwidth = zero
			}
		case "spreading_factor":
			if len(subs) > 0 {
				return fmt.Errorf("'spreading_factor' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.SpreadingFactor = src.SpreadingFactor
			} else {
				var zero uint32
				dst.SpreadingFactor = zero
			}
		case "data_rate_index":
			if len(subs) > 0 {
				return fmt.Errorf("'data_rate_index' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.DataRateIndex = src.DataRateIndex
			} else {
				var zero uint32
				dst.DataRateIndex = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *SimulateJoinRequestParams) SetFields(src *SimulateJoinRequestParams, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "join_eui":
			if len(subs) > 0 {
				return fmt.Errorf("'join_eui' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.JoinEui = src.JoinEui
			} else {
				dst.JoinEui = nil
			}
		case "dev_eui":
			if len(subs) > 0 {
				return fmt.Errorf("'dev_eui' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.DevEui = src.DevEui
			} else {
				dst.DevEui = nil
			}
		case "dev_nonce":
			if len(subs) > 0 {
				return fmt.Errorf("'dev_nonce' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.DevNonce = src.DevNonce
			} else {
				dst.DevNonce = nil
			}
		case "app_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.AppKey == nil) && dst.AppKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.AppKey
				}
				if dst.AppKey != nil {
					newDst = dst.AppKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.AppKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.AppKey = src.AppKey
				} else {
					dst.AppKey = nil
				}
			}
		case "nwk_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.NwkKey == nil) && dst.NwkKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.NwkKey
				}
				if dst.NwkKey != nil {
					newDst = dst.NwkKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.NwkKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.NwkKey = src.NwkKey
				} else {
					dst.NwkKey = nil
				}
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *SimulateDataUplinkParams) SetFields(src *SimulateDataUplinkParams, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "dev_addr":
			if len(subs) > 0 {
				return fmt.Errorf("'dev_addr' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.DevAddr = src.DevAddr
			} else {
				dst.DevAddr = nil
			}
		case "f_nwk_s_int_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.FNwkSIntKey == nil) && dst.FNwkSIntKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.FNwkSIntKey
				}
				if dst.FNwkSIntKey != nil {
					newDst = dst.FNwkSIntKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.FNwkSIntKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.FNwkSIntKey = src.FNwkSIntKey
				} else {
					dst.FNwkSIntKey = nil
				}
			}
		case "s_nwk_s_int_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.SNwkSIntKey == nil) && dst.SNwkSIntKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.SNwkSIntKey
				}
				if dst.SNwkSIntKey != nil {
					newDst = dst.SNwkSIntKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.SNwkSIntKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.SNwkSIntKey = src.SNwkSIntKey
				} else {
					dst.SNwkSIntKey = nil
				}
			}
		case "nwk_s_enc_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.NwkSEncKey == nil) && dst.NwkSEncKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.NwkSEncKey
				}
				if dst.NwkSEncKey != nil {
					newDst = dst.NwkSEncKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.NwkSEncKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.NwkSEncKey = src.NwkSEncKey
				} else {
					dst.NwkSEncKey = nil
				}
			}
		case "app_s_key":
			if len(subs) > 0 {
				var newDst, newSrc *ttnpb.KeyEnvelope
				if (src == nil || src.AppSKey == nil) && dst.AppSKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.AppSKey
				}
				if dst.AppSKey != nil {
					newDst = dst.AppSKey
				} else {
					newDst = &ttnpb.KeyEnvelope{}
					dst.AppSKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.AppSKey = src.AppSKey
				} else {
					dst.AppSKey = nil
				}
			}
		case "adr":
			if len(subs) > 0 {
				return fmt.Errorf("'adr' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Adr = src.Adr
			} else {
				var zero bool
				dst.Adr = zero
			}
		case "adr_ack_req":
			if len(subs) > 0 {
				return fmt.Errorf("'adr_ack_req' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.AdrAckReq = src.AdrAckReq
			} else {
				var zero bool
				dst.AdrAckReq = zero
			}
		case "confirmed":
			if len(subs) > 0 {
				return fmt.Errorf("'confirmed' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Confirmed = src.Confirmed
			} else {
				var zero bool
				dst.Confirmed = zero
			}
		case "ack":
			if len(subs) > 0 {
				return fmt.Errorf("'ack' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Ack = src.Ack
			} else {
				var zero bool
				dst.Ack = zero
			}
		case "f_cnt":
			if len(subs) > 0 {
				return fmt.Errorf("'f_cnt' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FCnt = src.FCnt
			} else {
				var zero uint32
				dst.FCnt = zero
			}
		case "f_port":
			if len(subs) > 0 {
				return fmt.Errorf("'f_port' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FPort = src.FPort
			} else {
				var zero uint32
				dst.FPort = zero
			}
		case "frm_payload":
			if len(subs) > 0 {
				return fmt.Errorf("'frm_payload' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FrmPayload = src.FrmPayload
			} else {
				dst.FrmPayload = nil
			}
		case "conf_f_cnt":
			if len(subs) > 0 {
				return fmt.Errorf("'conf_f_cnt' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ConfFCnt = src.ConfFCnt
			} else {
				var zero uint32
				dst.ConfFCnt = zero
			}
		case "tx_dr_idx":
			if len(subs) > 0 {
				return fmt.Errorf("'tx_dr_idx' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.TxDrIdx = src.TxDrIdx
			} else {
				var zero uint32
				dst.TxDrIdx = zero
			}
		case "tx_ch_idx":
			if len(subs) > 0 {
				return fmt.Errorf("'tx_ch_idx' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.TxChIdx = src.TxChIdx
			} else {
				var zero uint32
				dst.TxChIdx = zero
			}
		case "f_opts":
			if len(subs) > 0 {
				return fmt.Errorf("'f_opts' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FOpts = src.FOpts
			} else {
				dst.FOpts = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}
