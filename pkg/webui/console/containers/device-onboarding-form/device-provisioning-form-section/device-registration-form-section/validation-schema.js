// Copyright © 2022 The Things Network Foundation, The Things Industries B.V.
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

import Yup from '@ttn-lw/lib/yup'
import sharedMessages from '@ttn-lw/lib/shared-messages'

import { parseLorawanMacVersion } from '@console/lib/device-utils'

// Validation schema of the device registration form section.
// Please observe the following rules to keep the validation schemas maintainable:
// 1. DO NOT USE ANY TYPE CONVERSIONS HERE. Use decocer/encoder on field level instead.
//    Consider all values as backend values. Exceptions may apply in consideration.
// 2. Comment each individual validation prop and use whitespace to structure visually.
// 3. Do not use ternary assignments but use plain if statements to ensure clarity.

const devEUISchema = Yup.string().length(8 * 2, Yup.passValues(sharedMessages.validateLength))

const validationSchema = Yup.object({
  ids: Yup.object().shape({
    dev_eui: devEUISchema.when('$supportsJoin', {
      is: true,
      then: devEUISchema.required(sharedMessages.validateRequired),
    }),
    device_id: Yup.string(),
    join_eui: Yup.string()
      .length(8 * 2, Yup.passValues(sharedMessages.validateLength))
      .required(sharedMessages.validateRequired),
  }),

  root_keys: Yup.object().when(
    ['$lorawanVersion', '$supportsJoin', '$inputMethod', '$claim', '$mayEditKeys'],
    (isOTAA, version, inputMethod, claim, mayEditKeys, schema) => {
      const notRequiredKeySchema = Yup.object().shape({
        key: Yup.string().default('').nullable(),
      })
      const requiredKeySchema = Yup.lazy(() => {
        if (mayEditKeys) {
          return Yup.object().shape({
            key: Yup.string()
              .length(16 * 2, Yup.passValues(sharedMessages.validateLength))
              .required(sharedMessages.validateRequired),
          })
        }

        return notRequiredKeySchema
      })

      if (!mayEditKeys || !isOTAA || inputMethod === 'manual' || claim) {
        return schema.shape({
          nwk_key: notRequiredKeySchema,
          app_key: notRequiredKeySchema,
        })
      }

      if (parseLorawanMacVersion(version) < 110) {
        return schema.shape({
          nwk_key: notRequiredKeySchema,
          app_key: requiredKeySchema,
        })
      }

      return schema.shape({
        nwk_key: requiredKeySchema,
        app_key: requiredKeySchema,
      })
    },
  ),
  session: Yup.object().when(['$lorawanVersion', '$supportsJoin'], (version, isOTAA, schema) => {
    console.log(version)
    console.log(isOTAA)
    const lwVersion = parseLorawanMacVersion(version)
    const notRequiredKeySchema = Yup.object().shape({
      key: Yup.string().default('').nullable(),
    })

    return schema.shape({
      dev_addr: Yup.lazy(() => {
        if (!isOTAA) {
          return Yup.string()
            .length(4 * 2, Yup.passValues(sharedMessages.validateLength))
            .required(sharedMessages.validateRequired)
        }

        return notRequiredKeySchema
      }),
      keys: Yup.object().shape({
        app_s_key: Yup.lazy(() => {
          if (!isOTAA) {
            return Yup.object().shape({
              key: Yup.string()
                .length(16 * 2, Yup.passValues(sharedMessages.validateLength))
                .required(sharedMessages.validateRequired),
            })
          }

          return notRequiredKeySchema
        }),
        f_nwk_s_int_key: Yup.lazy(() => {
          if (!isOTAA) {
            return Yup.object().shape({
              key: Yup.string()
                .length(16 * 2, Yup.passValues(sharedMessages.validateLength))
                .required(sharedMessages.validateRequired),
            })
          }

          return notRequiredKeySchema
        }),
        s_nwk_s_int_key: Yup.lazy(() => {
          if (!isOTAA && lwVersion >= 110) {
            return Yup.object().shape({
              key: Yup.string()
                .length(16 * 2, Yup.passValues(sharedMessages.validateLength))
                .required(sharedMessages.validateRequired),
            })
          }

          return notRequiredKeySchema
        }),
        nwk_s_enc_key: Yup.lazy(() => {
          if (!isOTAA && lwVersion >= 110) {
            return Yup.object().shape({
              key: Yup.string()
                .length(16 * 2, Yup.passValues(sharedMessages.validateLength))
                .required(sharedMessages.validateRequired),
            })
          }

          return notRequiredKeySchema
        }),
      }),
    })
  }),
})

const initialValues = {
  ids: {
    device_id: '',
    dev_eui: '',
  },
  root_keys: {
    app_key: { key: '' },
    nwk_key: { key: '' },
  },
  session: {
    dev_addr: '',
    keys: {
      app_s_key: {
        key: '',
      },
      f_nwk_s_int_key: {
        key: '',
      },
      s_nwk_s_int_key: {
        key: '',
      },
      nwk_s_enc_key: {
        key: '',
      },
    },
  },
}

export { validationSchema as default, devEUISchema, initialValues }
