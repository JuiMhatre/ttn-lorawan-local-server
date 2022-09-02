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

// Parse the QR code using the LoRa Alliance TR005 Draft 3 and final specifications.
// See https://lora-alliance.org/wp-content/uploads/2020/11/TR005_LoRaWAN_Device_Identification_QR_Codes.pdf
// If the prefix is not recognized, this function returns undefined.
// eslint-disable-next-line import/prefer-default-export
export const readDeviceQr = qrCode => {
  if (Boolean(qrCode.match(/^LW:D0:/))) {
    const parts = qrCode.split(':')
    return {
      formatId: 'tr005',
      schemaId: parts[1],
      joinEUI: parts[2],
      devEUI: parts[3],
      profileID: {
        vendorID: parts[4].substring(0, 4),
        vendorProfileID: parts[4].substring(4, 8),
      },
      ownerToken: parts[5],
      qrCode,
    }
  } else if (Boolean(qrCode.match(/^URN:DEV:LW:/))) {
    const parts = qrCode.split(':')[3].split('_')
    return {
      formatId: 'tr005draft3',
      joinEUI: parts[0],
      devEUI: parts[1],
      profileID: {
        vendorID: parts[2].substring(0, 4),
        vendorProfileID: parts[2].substring(4, 8),
      },
      ownerToken: parts[3],
      qrCode,
    }
  } else if (Boolean(qrCode.match(/^URN:LW:DP:/))) {
    const parts = qrCode.split(':')[3].split('_')
    return {
      formatId: 'tr005draft2',
      joinEUI: parts[0],
      devEUI: parts[1],
      profileID: {
        vendorID: parts[2].substring(0, 4),
        vendorProfileID: parts[2].substring(4, 8),
      },
      qrCode,
    }
  }
}
