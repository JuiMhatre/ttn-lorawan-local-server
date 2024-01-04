// Code generated by protoc-gen-fieldmask. DO NOT EDIT.

package ttnpb

import fmt "fmt"

func (dst *UserConsolePreferences) SetFields(src *UserConsolePreferences, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "console_theme":
			if len(subs) > 0 {
				return fmt.Errorf("'console_theme' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ConsoleTheme = src.ConsoleTheme
			} else {
				dst.ConsoleTheme = 0
			}
		case "dashboard_layouts":
			if len(subs) > 0 {
				var newDst, newSrc *UserConsolePreferences_DashboardLayouts
				if (src == nil || src.DashboardLayouts == nil) && dst.DashboardLayouts == nil {
					continue
				}
				if src != nil {
					newSrc = src.DashboardLayouts
				}
				if dst.DashboardLayouts != nil {
					newDst = dst.DashboardLayouts
				} else {
					newDst = &UserConsolePreferences_DashboardLayouts{}
					dst.DashboardLayouts = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.DashboardLayouts = src.DashboardLayouts
				} else {
					dst.DashboardLayouts = nil
				}
			}
		case "sort_by":
			if len(subs) > 0 {
				var newDst, newSrc *UserConsolePreferences_SortBy
				if (src == nil || src.SortBy == nil) && dst.SortBy == nil {
					continue
				}
				if src != nil {
					newSrc = src.SortBy
				}
				if dst.SortBy != nil {
					newDst = dst.SortBy
				} else {
					newDst = &UserConsolePreferences_SortBy{}
					dst.SortBy = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.SortBy = src.SortBy
				} else {
					dst.SortBy = nil
				}
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *User) SetFields(src *User, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.Ids == nil) && dst.Ids == nil {
					continue
				}
				if src != nil {
					newSrc = src.Ids
				}
				if dst.Ids != nil {
					newDst = dst.Ids
				} else {
					newDst = &UserIdentifiers{}
					dst.Ids = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.Ids = src.Ids
				} else {
					dst.Ids = nil
				}
			}
		case "created_at":
			if len(subs) > 0 {
				return fmt.Errorf("'created_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.CreatedAt = src.CreatedAt
			} else {
				dst.CreatedAt = nil
			}
		case "updated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'updated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.UpdatedAt = src.UpdatedAt
			} else {
				dst.UpdatedAt = nil
			}
		case "deleted_at":
			if len(subs) > 0 {
				return fmt.Errorf("'deleted_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.DeletedAt = src.DeletedAt
			} else {
				dst.DeletedAt = nil
			}
		case "name":
			if len(subs) > 0 {
				return fmt.Errorf("'name' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Name = src.Name
			} else {
				var zero string
				dst.Name = zero
			}
		case "description":
			if len(subs) > 0 {
				return fmt.Errorf("'description' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Description = src.Description
			} else {
				var zero string
				dst.Description = zero
			}
		case "attributes":
			if len(subs) > 0 {
				return fmt.Errorf("'attributes' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Attributes = src.Attributes
			} else {
				dst.Attributes = nil
			}
		case "contact_info":
			if len(subs) > 0 {
				return fmt.Errorf("'contact_info' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ContactInfo = src.ContactInfo
			} else {
				dst.ContactInfo = nil
			}
		case "primary_email_address":
			if len(subs) > 0 {
				return fmt.Errorf("'primary_email_address' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.PrimaryEmailAddress = src.PrimaryEmailAddress
			} else {
				var zero string
				dst.PrimaryEmailAddress = zero
			}
		case "primary_email_address_validated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'primary_email_address_validated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.PrimaryEmailAddressValidatedAt = src.PrimaryEmailAddressValidatedAt
			} else {
				dst.PrimaryEmailAddressValidatedAt = nil
			}
		case "password":
			if len(subs) > 0 {
				return fmt.Errorf("'password' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Password = src.Password
			} else {
				var zero string
				dst.Password = zero
			}
		case "password_updated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'password_updated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.PasswordUpdatedAt = src.PasswordUpdatedAt
			} else {
				dst.PasswordUpdatedAt = nil
			}
		case "require_password_update":
			if len(subs) > 0 {
				return fmt.Errorf("'require_password_update' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.RequirePasswordUpdate = src.RequirePasswordUpdate
			} else {
				var zero bool
				dst.RequirePasswordUpdate = zero
			}
		case "state":
			if len(subs) > 0 {
				return fmt.Errorf("'state' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.State = src.State
			} else {
				dst.State = 0
			}
		case "state_description":
			if len(subs) > 0 {
				return fmt.Errorf("'state_description' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.StateDescription = src.StateDescription
			} else {
				var zero string
				dst.StateDescription = zero
			}
		case "admin":
			if len(subs) > 0 {
				return fmt.Errorf("'admin' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Admin = src.Admin
			} else {
				var zero bool
				dst.Admin = zero
			}
		case "temporary_password":
			if len(subs) > 0 {
				return fmt.Errorf("'temporary_password' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.TemporaryPassword = src.TemporaryPassword
			} else {
				var zero string
				dst.TemporaryPassword = zero
			}
		case "temporary_password_created_at":
			if len(subs) > 0 {
				return fmt.Errorf("'temporary_password_created_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.TemporaryPasswordCreatedAt = src.TemporaryPasswordCreatedAt
			} else {
				dst.TemporaryPasswordCreatedAt = nil
			}
		case "temporary_password_expires_at":
			if len(subs) > 0 {
				return fmt.Errorf("'temporary_password_expires_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.TemporaryPasswordExpiresAt = src.TemporaryPasswordExpiresAt
			} else {
				dst.TemporaryPasswordExpiresAt = nil
			}
		case "profile_picture":
			if len(subs) > 0 {
				var newDst, newSrc *Picture
				if (src == nil || src.ProfilePicture == nil) && dst.ProfilePicture == nil {
					continue
				}
				if src != nil {
					newSrc = src.ProfilePicture
				}
				if dst.ProfilePicture != nil {
					newDst = dst.ProfilePicture
				} else {
					newDst = &Picture{}
					dst.ProfilePicture = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.ProfilePicture = src.ProfilePicture
				} else {
					dst.ProfilePicture = nil
				}
			}
		case "console_preferences":
			if len(subs) > 0 {
				var newDst, newSrc *UserConsolePreferences
				if (src == nil || src.ConsolePreferences == nil) && dst.ConsolePreferences == nil {
					continue
				}
				if src != nil {
					newSrc = src.ConsolePreferences
				}
				if dst.ConsolePreferences != nil {
					newDst = dst.ConsolePreferences
				} else {
					newDst = &UserConsolePreferences{}
					dst.ConsolePreferences = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.ConsolePreferences = src.ConsolePreferences
				} else {
					dst.ConsolePreferences = nil
				}
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *Users) SetFields(src *Users, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "users":
			if len(subs) > 0 {
				return fmt.Errorf("'users' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Users = src.Users
			} else {
				dst.Users = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *GetUserRequest) SetFields(src *GetUserRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "field_mask":
			if len(subs) > 0 {
				return fmt.Errorf("'field_mask' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FieldMask = src.FieldMask
			} else {
				dst.FieldMask = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *ListUsersRequest) SetFields(src *ListUsersRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "field_mask":
			if len(subs) > 0 {
				return fmt.Errorf("'field_mask' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FieldMask = src.FieldMask
			} else {
				dst.FieldMask = nil
			}
		case "order":
			if len(subs) > 0 {
				return fmt.Errorf("'order' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Order = src.Order
			} else {
				var zero string
				dst.Order = zero
			}
		case "limit":
			if len(subs) > 0 {
				return fmt.Errorf("'limit' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Limit = src.Limit
			} else {
				var zero uint32
				dst.Limit = zero
			}
		case "page":
			if len(subs) > 0 {
				return fmt.Errorf("'page' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Page = src.Page
			} else {
				var zero uint32
				dst.Page = zero
			}
		case "deleted":
			if len(subs) > 0 {
				return fmt.Errorf("'deleted' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Deleted = src.Deleted
			} else {
				var zero bool
				dst.Deleted = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *CreateUserRequest) SetFields(src *CreateUserRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user":
			if len(subs) > 0 {
				var newDst, newSrc *User
				if (src == nil || src.User == nil) && dst.User == nil {
					continue
				}
				if src != nil {
					newSrc = src.User
				}
				if dst.User != nil {
					newDst = dst.User
				} else {
					newDst = &User{}
					dst.User = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.User = src.User
				} else {
					dst.User = nil
				}
			}
		case "invitation_token":
			if len(subs) > 0 {
				return fmt.Errorf("'invitation_token' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.InvitationToken = src.InvitationToken
			} else {
				var zero string
				dst.InvitationToken = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UpdateUserRequest) SetFields(src *UpdateUserRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user":
			if len(subs) > 0 {
				var newDst, newSrc *User
				if (src == nil || src.User == nil) && dst.User == nil {
					continue
				}
				if src != nil {
					newSrc = src.User
				}
				if dst.User != nil {
					newDst = dst.User
				} else {
					newDst = &User{}
					dst.User = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.User = src.User
				} else {
					dst.User = nil
				}
			}
		case "field_mask":
			if len(subs) > 0 {
				return fmt.Errorf("'field_mask' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FieldMask = src.FieldMask
			} else {
				dst.FieldMask = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *CreateTemporaryPasswordRequest) SetFields(src *CreateTemporaryPasswordRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UpdateUserPasswordRequest) SetFields(src *UpdateUserPasswordRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "new":
			if len(subs) > 0 {
				return fmt.Errorf("'new' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.New = src.New
			} else {
				var zero string
				dst.New = zero
			}
		case "old":
			if len(subs) > 0 {
				return fmt.Errorf("'old' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Old = src.Old
			} else {
				var zero string
				dst.Old = zero
			}
		case "revoke_all_access":
			if len(subs) > 0 {
				return fmt.Errorf("'revoke_all_access' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.RevokeAllAccess = src.RevokeAllAccess
			} else {
				var zero bool
				dst.RevokeAllAccess = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *ListUserAPIKeysRequest) SetFields(src *ListUserAPIKeysRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "order":
			if len(subs) > 0 {
				return fmt.Errorf("'order' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Order = src.Order
			} else {
				var zero string
				dst.Order = zero
			}
		case "limit":
			if len(subs) > 0 {
				return fmt.Errorf("'limit' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Limit = src.Limit
			} else {
				var zero uint32
				dst.Limit = zero
			}
		case "page":
			if len(subs) > 0 {
				return fmt.Errorf("'page' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Page = src.Page
			} else {
				var zero uint32
				dst.Page = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *GetUserAPIKeyRequest) SetFields(src *GetUserAPIKeyRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "key_id":
			if len(subs) > 0 {
				return fmt.Errorf("'key_id' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.KeyId = src.KeyId
			} else {
				var zero string
				dst.KeyId = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *CreateUserAPIKeyRequest) SetFields(src *CreateUserAPIKeyRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "name":
			if len(subs) > 0 {
				return fmt.Errorf("'name' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Name = src.Name
			} else {
				var zero string
				dst.Name = zero
			}
		case "rights":
			if len(subs) > 0 {
				return fmt.Errorf("'rights' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Rights = src.Rights
			} else {
				dst.Rights = nil
			}
		case "expires_at":
			if len(subs) > 0 {
				return fmt.Errorf("'expires_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ExpiresAt = src.ExpiresAt
			} else {
				dst.ExpiresAt = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UpdateUserAPIKeyRequest) SetFields(src *UpdateUserAPIKeyRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "api_key":
			if len(subs) > 0 {
				var newDst, newSrc *APIKey
				if (src == nil || src.ApiKey == nil) && dst.ApiKey == nil {
					continue
				}
				if src != nil {
					newSrc = src.ApiKey
				}
				if dst.ApiKey != nil {
					newDst = dst.ApiKey
				} else {
					newDst = &APIKey{}
					dst.ApiKey = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.ApiKey = src.ApiKey
				} else {
					dst.ApiKey = nil
				}
			}
		case "field_mask":
			if len(subs) > 0 {
				return fmt.Errorf("'field_mask' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.FieldMask = src.FieldMask
			} else {
				dst.FieldMask = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *DeleteUserAPIKeyRequest) SetFields(src *DeleteUserAPIKeyRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "key_id":
			if len(subs) > 0 {
				return fmt.Errorf("'key_id' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.KeyId = src.KeyId
			} else {
				var zero string
				dst.KeyId = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *Invitation) SetFields(src *Invitation, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "email":
			if len(subs) > 0 {
				return fmt.Errorf("'email' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Email = src.Email
			} else {
				var zero string
				dst.Email = zero
			}
		case "token":
			if len(subs) > 0 {
				return fmt.Errorf("'token' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Token = src.Token
			} else {
				var zero string
				dst.Token = zero
			}
		case "expires_at":
			if len(subs) > 0 {
				return fmt.Errorf("'expires_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ExpiresAt = src.ExpiresAt
			} else {
				dst.ExpiresAt = nil
			}
		case "created_at":
			if len(subs) > 0 {
				return fmt.Errorf("'created_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.CreatedAt = src.CreatedAt
			} else {
				dst.CreatedAt = nil
			}
		case "updated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'updated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.UpdatedAt = src.UpdatedAt
			} else {
				dst.UpdatedAt = nil
			}
		case "accepted_at":
			if len(subs) > 0 {
				return fmt.Errorf("'accepted_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.AcceptedAt = src.AcceptedAt
			} else {
				dst.AcceptedAt = nil
			}
		case "accepted_by":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.AcceptedBy == nil) && dst.AcceptedBy == nil {
					continue
				}
				if src != nil {
					newSrc = src.AcceptedBy
				}
				if dst.AcceptedBy != nil {
					newDst = dst.AcceptedBy
				} else {
					newDst = &UserIdentifiers{}
					dst.AcceptedBy = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.AcceptedBy = src.AcceptedBy
				} else {
					dst.AcceptedBy = nil
				}
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *ListInvitationsRequest) SetFields(src *ListInvitationsRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "limit":
			if len(subs) > 0 {
				return fmt.Errorf("'limit' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Limit = src.Limit
			} else {
				var zero uint32
				dst.Limit = zero
			}
		case "page":
			if len(subs) > 0 {
				return fmt.Errorf("'page' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Page = src.Page
			} else {
				var zero uint32
				dst.Page = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *Invitations) SetFields(src *Invitations, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "invitations":
			if len(subs) > 0 {
				return fmt.Errorf("'invitations' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Invitations = src.Invitations
			} else {
				dst.Invitations = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *SendInvitationRequest) SetFields(src *SendInvitationRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "email":
			if len(subs) > 0 {
				return fmt.Errorf("'email' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Email = src.Email
			} else {
				var zero string
				dst.Email = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *DeleteInvitationRequest) SetFields(src *DeleteInvitationRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "email":
			if len(subs) > 0 {
				return fmt.Errorf("'email' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Email = src.Email
			} else {
				var zero string
				dst.Email = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UserSessionIdentifiers) SetFields(src *UserSessionIdentifiers, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "session_id":
			if len(subs) > 0 {
				return fmt.Errorf("'session_id' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.SessionId = src.SessionId
			} else {
				var zero string
				dst.SessionId = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UserSession) SetFields(src *UserSession, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "session_id":
			if len(subs) > 0 {
				return fmt.Errorf("'session_id' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.SessionId = src.SessionId
			} else {
				var zero string
				dst.SessionId = zero
			}
		case "created_at":
			if len(subs) > 0 {
				return fmt.Errorf("'created_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.CreatedAt = src.CreatedAt
			} else {
				dst.CreatedAt = nil
			}
		case "updated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'updated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.UpdatedAt = src.UpdatedAt
			} else {
				dst.UpdatedAt = nil
			}
		case "expires_at":
			if len(subs) > 0 {
				return fmt.Errorf("'expires_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ExpiresAt = src.ExpiresAt
			} else {
				dst.ExpiresAt = nil
			}
		case "session_secret":
			if len(subs) > 0 {
				return fmt.Errorf("'session_secret' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.SessionSecret = src.SessionSecret
			} else {
				var zero string
				dst.SessionSecret = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UserSessions) SetFields(src *UserSessions, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "sessions":
			if len(subs) > 0 {
				return fmt.Errorf("'sessions' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Sessions = src.Sessions
			} else {
				dst.Sessions = nil
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *ListUserSessionsRequest) SetFields(src *ListUserSessionsRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "order":
			if len(subs) > 0 {
				return fmt.Errorf("'order' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Order = src.Order
			} else {
				var zero string
				dst.Order = zero
			}
		case "limit":
			if len(subs) > 0 {
				return fmt.Errorf("'limit' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Limit = src.Limit
			} else {
				var zero uint32
				dst.Limit = zero
			}
		case "page":
			if len(subs) > 0 {
				return fmt.Errorf("'page' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Page = src.Page
			} else {
				var zero uint32
				dst.Page = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *LoginToken) SetFields(src *LoginToken, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "created_at":
			if len(subs) > 0 {
				return fmt.Errorf("'created_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.CreatedAt = src.CreatedAt
			} else {
				dst.CreatedAt = nil
			}
		case "updated_at":
			if len(subs) > 0 {
				return fmt.Errorf("'updated_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.UpdatedAt = src.UpdatedAt
			} else {
				dst.UpdatedAt = nil
			}
		case "expires_at":
			if len(subs) > 0 {
				return fmt.Errorf("'expires_at' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ExpiresAt = src.ExpiresAt
			} else {
				dst.ExpiresAt = nil
			}
		case "token":
			if len(subs) > 0 {
				return fmt.Errorf("'token' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Token = src.Token
			} else {
				var zero string
				dst.Token = zero
			}
		case "used":
			if len(subs) > 0 {
				return fmt.Errorf("'used' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Used = src.Used
			} else {
				var zero bool
				dst.Used = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *CreateLoginTokenRequest) SetFields(src *CreateLoginTokenRequest, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "user_ids":
			if len(subs) > 0 {
				var newDst, newSrc *UserIdentifiers
				if (src == nil || src.UserIds == nil) && dst.UserIds == nil {
					continue
				}
				if src != nil {
					newSrc = src.UserIds
				}
				if dst.UserIds != nil {
					newDst = dst.UserIds
				} else {
					newDst = &UserIdentifiers{}
					dst.UserIds = newDst
				}
				if err := newDst.SetFields(newSrc, subs...); err != nil {
					return err
				}
			} else {
				if src != nil {
					dst.UserIds = src.UserIds
				} else {
					dst.UserIds = nil
				}
			}
		case "skip_email":
			if len(subs) > 0 {
				return fmt.Errorf("'skip_email' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.SkipEmail = src.SkipEmail
			} else {
				var zero bool
				dst.SkipEmail = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *CreateLoginTokenResponse) SetFields(src *CreateLoginTokenResponse, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "token":
			if len(subs) > 0 {
				return fmt.Errorf("'token' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Token = src.Token
			} else {
				var zero string
				dst.Token = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UserConsolePreferences_DashboardLayouts) SetFields(src *UserConsolePreferences_DashboardLayouts, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "api_key":
			if len(subs) > 0 {
				return fmt.Errorf("'api_key' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ApiKey = src.ApiKey
			} else {
				dst.ApiKey = 0
			}
		case "application":
			if len(subs) > 0 {
				return fmt.Errorf("'application' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Application = src.Application
			} else {
				dst.Application = 0
			}
		case "collaborator":
			if len(subs) > 0 {
				return fmt.Errorf("'collaborator' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Collaborator = src.Collaborator
			} else {
				dst.Collaborator = 0
			}
		case "end_device":
			if len(subs) > 0 {
				return fmt.Errorf("'end_device' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.EndDevice = src.EndDevice
			} else {
				dst.EndDevice = 0
			}
		case "gateway":
			if len(subs) > 0 {
				return fmt.Errorf("'gateway' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Gateway = src.Gateway
			} else {
				dst.Gateway = 0
			}
		case "organization":
			if len(subs) > 0 {
				return fmt.Errorf("'organization' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Organization = src.Organization
			} else {
				dst.Organization = 0
			}
		case "overview":
			if len(subs) > 0 {
				return fmt.Errorf("'overview' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Overview = src.Overview
			} else {
				dst.Overview = 0
			}
		case "user":
			if len(subs) > 0 {
				return fmt.Errorf("'user' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.User = src.User
			} else {
				dst.User = 0
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}

func (dst *UserConsolePreferences_SortBy) SetFields(src *UserConsolePreferences_SortBy, paths ...string) error {
	for name, subs := range _processPaths(paths) {
		switch name {
		case "api_key":
			if len(subs) > 0 {
				return fmt.Errorf("'api_key' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.ApiKey = src.ApiKey
			} else {
				var zero string
				dst.ApiKey = zero
			}
		case "application":
			if len(subs) > 0 {
				return fmt.Errorf("'application' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Application = src.Application
			} else {
				var zero string
				dst.Application = zero
			}
		case "collaborator":
			if len(subs) > 0 {
				return fmt.Errorf("'collaborator' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Collaborator = src.Collaborator
			} else {
				var zero string
				dst.Collaborator = zero
			}
		case "end_device":
			if len(subs) > 0 {
				return fmt.Errorf("'end_device' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.EndDevice = src.EndDevice
			} else {
				var zero string
				dst.EndDevice = zero
			}
		case "gateway":
			if len(subs) > 0 {
				return fmt.Errorf("'gateway' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Gateway = src.Gateway
			} else {
				var zero string
				dst.Gateway = zero
			}
		case "organization":
			if len(subs) > 0 {
				return fmt.Errorf("'organization' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.Organization = src.Organization
			} else {
				var zero string
				dst.Organization = zero
			}
		case "user":
			if len(subs) > 0 {
				return fmt.Errorf("'user' has no subfields, but %s were specified", subs)
			}
			if src != nil {
				dst.User = src.User
			} else {
				var zero string
				dst.User = zero
			}

		default:
			return fmt.Errorf("invalid field: '%s'", name)
		}
	}
	return nil
}
