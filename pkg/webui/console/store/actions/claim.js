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

import createRequestActions from '@ttn-lw/lib/store/actions/create-request-actions'

const CLAIM_DEVICE_BASE = 'CLAIM_DEVICE'
export const [
  { request: CLAIM_DEVICE, success: CLAIM_DEVICE_SUCCESS, failure: CLAIM_DEVICE_FAILURE },
  { request: claimDevice, success: claimDeviceSuccess, failure: claimDeviceFailure },
] = createRequestActions(CLAIM_DEVICE_BASE, (appId, qr_code, authenticatedIdentifiers) => ({
  appId,
  qr_code,
  authenticatedIdentifiers,
}))

const UNCLAIM_DEVICE_BASE = 'UNCLAIM_DEVICE'
export const [
  { request: UNCLAIM_DEVICE, success: UNCLAIM_DEVICE_SUCCESS, failure: UNCLAIM_DEVICE_FAILURE },
  { request: unclaimDevice, success: unclaimDeviceSuccess, failure: unclaimDeviceFailure },
] = createRequestActions(UNCLAIM_DEVICE_BASE, (applicationId, deviceId) => ({
  applicationId,
  deviceId,
}))

const GET_INFO_BY_JOIN_EUI_BASE = 'GET_INFO_BY_JOIN_EUI'
export const [
  {
    request: GET_INFO_BY_JOIN_EUI,
    success: GET_INFO_BY_JOIN_EUI_SUCCESS,
    failure: GET_INFO_BY_JOIN_EUI_FAILURE,
  },
  { request: getInfoByJoinEUI, success: getInfoByJoinEUISuccess, failure: getInfoByJoinEUIFailure },
] = createRequestActions(GET_INFO_BY_JOIN_EUI_BASE, joinEUI => ({ joinEUI }))
