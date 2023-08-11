// Code generated by protoc-gen-go-flags. DO NOT EDIT.
// versions:
// - protoc-gen-go-flags v1.1.0
// - protoc              v4.22.2
// source: ttn/lorawan/v3/contact_info.proto

package ttnpb

import (
	flagsplugin "github.com/TheThingsIndustries/protoc-gen-go-flags/flagsplugin"
	golang "github.com/TheThingsIndustries/protoc-gen-go-flags/golang"
	pflag "github.com/spf13/pflag"
)

// AddSelectFlagsForContactInfo adds flags to select fields in ContactInfo.
func AddSelectFlagsForContactInfo(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("contact-type", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("contact-type", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("contact-method", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("contact-method", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("value", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("value", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("public", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("public", prefix), false), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("validated-at", prefix), flagsplugin.SelectDesc(flagsplugin.Prefix("validated-at", prefix), false), flagsplugin.WithHidden(hidden)))
}

// SelectFromFlags outputs the fieldmask paths forContactInfo message from select flags.
func PathsFromSelectFlagsForContactInfo(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("contact_type", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("contact_type", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("contact_method", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("contact_method", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("value", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("value", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("public", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("public", prefix))
	}
	if val, selected, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("validated_at", prefix)); err != nil {
		return nil, err
	} else if selected && val {
		paths = append(paths, flagsplugin.Prefix("validated_at", prefix))
	}
	return paths, nil
}

// AddSetFlagsForContactInfo adds flags to select fields in ContactInfo.
func AddSetFlagsForContactInfo(flags *pflag.FlagSet, prefix string, hidden bool) {
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("contact-type", prefix), flagsplugin.EnumValueDesc(ContactType_value), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("contact-method", prefix), flagsplugin.EnumValueDesc(ContactMethod_value), flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewStringFlag(flagsplugin.Prefix("value", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewBoolFlag(flagsplugin.Prefix("public", prefix), "", flagsplugin.WithHidden(hidden)))
	flags.AddFlag(flagsplugin.NewTimestampFlag(flagsplugin.Prefix("validated-at", prefix), "", flagsplugin.WithHidden(hidden)))
}

// SetFromFlags sets the ContactInfo message from flags.
func (m *ContactInfo) SetFromFlags(flags *pflag.FlagSet, prefix string) (paths []string, err error) {
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("contact_type", prefix)); err != nil {
		return nil, err
	} else if changed {
		enumValue, err := flagsplugin.SetEnumString(val, ContactType_value)
		if err != nil {
			return nil, err
		}
		m.ContactType = ContactType(enumValue)
		paths = append(paths, flagsplugin.Prefix("contact_type", prefix))
	}
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("contact_method", prefix)); err != nil {
		return nil, err
	} else if changed {
		enumValue, err := flagsplugin.SetEnumString(val, ContactMethod_value)
		if err != nil {
			return nil, err
		}
		m.ContactMethod = ContactMethod(enumValue)
		paths = append(paths, flagsplugin.Prefix("contact_method", prefix))
	}
	if val, changed, err := flagsplugin.GetString(flags, flagsplugin.Prefix("value", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Value = val
		paths = append(paths, flagsplugin.Prefix("value", prefix))
	}
	if val, changed, err := flagsplugin.GetBool(flags, flagsplugin.Prefix("public", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.Public = val
		paths = append(paths, flagsplugin.Prefix("public", prefix))
	}
	if val, changed, err := flagsplugin.GetTimestamp(flags, flagsplugin.Prefix("validated_at", prefix)); err != nil {
		return nil, err
	} else if changed {
		m.ValidatedAt = golang.SetTimestamp(val)
		paths = append(paths, flagsplugin.Prefix("validated_at", prefix))
	}
	return paths, nil
}
