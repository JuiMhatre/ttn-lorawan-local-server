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

import React from 'react'

import Input from '@ttn-lw/components/input'
import Form from '@ttn-lw/components/form'

import DevEUIComponent from '@console/components/dev-eui-component'

import sharedMessages from '@ttn-lw/lib/shared-messages'

const initialValues = {
  ids: {
    dev_eui: '',
  },
  authentication_code: '',
}

const DeviceClaimingFormSection = () => (
  <>
    <DevEUIComponent />
    <Form.Field
      title={sharedMessages.claimAuthCode}
      name="authentication_code"
      component={Input}
      sensitive
    />
  </>
)

export { DeviceClaimingFormSection as default, initialValues }
