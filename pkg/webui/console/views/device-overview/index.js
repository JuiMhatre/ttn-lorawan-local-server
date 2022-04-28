// Copyright © 2019 The Things Network Foundation, The Things Industries B.V.
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
import bind from 'autobind-decorator'
import { connect } from 'react-redux'
import { Col, Row, Container } from 'react-grid-system'
import { defineMessages } from 'react-intl'

import tts from '@console/api/tts'

import DataSheet from '@ttn-lw/components/data-sheet'
import ModalButton from '@ttn-lw/components/button/modal-button'

import IntlHelmet from '@ttn-lw/lib/components/intl-helmet'
import Message from '@ttn-lw/lib/components/message'
import DateTime from '@ttn-lw/lib/components/date-time'

import DeviceMap from '@console/components/device-map'

import DeviceEvents from '@console/containers/device-events'

import Require from '@console/lib/components/require'

import sharedMessages from '@ttn-lw/lib/shared-messages'
import PropTypes from '@ttn-lw/lib/prop-types'
import { composeDataUri, downloadDataUriAsFile } from '@ttn-lw/lib/data-uri'

import { parseLorawanMacVersion } from '@console/lib/device-utils'

import { selectSelectedDevice, isOtherClusterDevice } from '@console/store/selectors/devices'
import { selectSelectedApplicationId } from '@console/store/selectors/applications'

import style from './device-overview.styl'

const m = defineMessages({
  activationInfo: 'Activation information',
  rootKeyId: 'Root key ID',
  sessionInfo: 'Session information',
  latestData: 'Latest data',
  rootKeys: 'Root keys',
  keysNotExposed: 'Keys are not exposed',
  failedAccessOtherHostDevice:
    'The end device you attempted to visit is registered on a different cluster and needs to be accessed using its host Console.',
  macData: 'Download MAC data',
  hasSession:
    'The MAC data can contain sensitive information such as session keys that can be used to decrypt messages. Do not share this information publicly.',
  noSession:
    'The end device is currently not connected to the network (no active session). The MAC data will hence only contain the current MAC settings.',
})

@connect(state => {
  const appId = selectSelectedApplicationId(state)
  const device = selectSelectedDevice(state)
  const shouldRedirect = isOtherClusterDevice(device)
  return {
    appId,
    device,
    shouldRedirect,
  }
})
class DeviceOverview extends React.Component {
  static propTypes = {
    appId: PropTypes.string.isRequired,
    device: PropTypes.device.isRequired,
    shouldRedirect: PropTypes.bool,
  }

  static defaultProps = {
    shouldRedirect: false,
  }

  @bind
  async onExport() {
    const {
      appId,
      device: { ids, mac_settings },
    } = this.props
    const result = await tts.Applications.Devices.getById(appId, ids.device_id, ['mac_state'])
    const toExport = { mac_state: result.mac_state, mac_settings }
    const toExportData = composeDataUri(JSON.stringify(toExport, undefined, 2))
    downloadDataUriAsFile(toExportData, `${ids.device_id}_mac_data_${Date.now()}.json`)
  }

  get deviceInfo() {
    const {
      ids,
      description,
      version_ids = {},
      root_keys = {},
      session = {},
      created_at,
      lorawan_version,
      supports_join,
      mac_state,
      mac_settings,
    } = this.props.device

    // Get session keys.
    const { keys: sessionKeys = {} } = session

    const lorawanVersion = parseLorawanMacVersion(lorawan_version)

    const {
      f_nwk_s_int_key = { key: undefined },
      s_nwk_s_int_key = { key: undefined },
      nwk_s_enc_key = { key: undefined },
      app_s_key = { key: undefined },
    } = sessionKeys

    const sheetData = [
      {
        header: sharedMessages.generalInformation,
        items: [
          { key: sharedMessages.devID, value: ids.device_id, type: 'code', sensitive: false },
          {
            key: sharedMessages.description,
            value: description || <Message content={sharedMessages.noDesc} />,
          },
          { key: sharedMessages.createdAt, value: <DateTime value={created_at} /> },
        ],
      },
    ]

    // Add version info, if it is available.
    if (Object.keys(version_ids).length > 0) {
      sheetData.push({
        header: sharedMessages.hardware,
        items: [
          { key: sharedMessages.brand, value: version_ids.brand_id },
          { key: sharedMessages.model, value: version_ids.model_id },
          { key: sharedMessages.hardwareVersion, value: version_ids.hardware_version },
          { key: sharedMessages.firmwareVersion, value: version_ids.firmware_version },
        ],
      })
    }

    // Add activation info, if available.
    const activationInfoData = {
      header: m.activationInfo,
      items: [],
    }

    if (ids.join_eui || ids.dev_eui) {
      const joinEUI =
        lorawanVersion < 100
          ? sharedMessages.appEUIJoinEUI
          : lorawanVersion >= 104
          ? sharedMessages.joinEUI
          : sharedMessages.appEUI

      activationInfoData.items.push(
        { key: joinEUI, value: ids.join_eui, type: 'byte', sensitive: false },
        { key: sharedMessages.devEUI, value: ids.dev_eui, type: 'byte', sensitive: false },
      )

      // Add root keys, if available.
      if (Object.keys(root_keys).length > 0) {
        const infoEntry = {
          key: m.rootKeyId,
          value: root_keys.root_key_id,
          type: 'code',
          sensitive: false,
        }
        if (!Boolean(root_keys.app_key) && !Boolean(root_keys.nwk_key)) {
          infoEntry.subItems = [
            {
              key: m.rootKeys,
              value: <Message content={m.keysNotExposed} />,
            },
          ]
        } else {
          infoEntry.subItems = [
            ...(root_keys.app_key
              ? {
                  key: sharedMessages.appKey,
                  value: root_keys.app_key.key,
                  type: 'byte',
                  sensitive: true,
                }
              : { key: sharedMessages.appKey, value: undefined }),
            ...(root_keys.nwk_key
              ? {
                  key: sharedMessages.nwkKey,
                  value: root_keys.nwk_key.key,
                  type: 'byte',
                  sensitive: true,
                }
              : { key: sharedMessages.nwkKey, value: undefined }),
          ]
        }
        activationInfoData.items.push(infoEntry)
      } else if (supports_join) {
        activationInfoData.items.push({
          key: m.rootKeys,
          value: <Message content={sharedMessages.provisionedOnExternalJoinServer} />,
        })
      }
    }

    sheetData.push(activationInfoData)

    // Add session info, if available.

    const sessionInfoData = {
      header: m.sessionInfo,
      items: [],
    }

    if (Object.keys(sessionKeys).length > 0) {
      sessionInfoData.items.push(
        {
          key: sharedMessages.devAddr,
          value: ids.dev_addr,
          type: 'byte',
          sensitive: false,
          enableUint32: true,
        },
        {
          key: sharedMessages.nwkSKey,
          value: f_nwk_s_int_key.key,
          type: 'byte',
          sensitive: true,
        },
        {
          key: sharedMessages.sNwkSIKey,
          value: s_nwk_s_int_key.key,
          type: 'byte',
          sensitive: true,
        },
        {
          key: sharedMessages.nwkSEncKey,
          value: nwk_s_enc_key.key,
          type: 'byte',
          sensitive: true,
        },
        { key: sharedMessages.appSKey, value: app_s_key.key, type: 'byte', sensitive: true },
      )
    }

    sheetData.push(sessionInfoData)

    const macStateAndSettings = {
      header: 'MAC data',
      items: [],
    }

    if (Boolean(mac_state) && Boolean(mac_settings)) {
      macStateAndSettings.items.push({
        value: (
          <ModalButton
            modalData={{
              message: session ? m.hasSession : m.noSession,
            }}
            onApprove={this.onExport}
            message={m.macData}
            type="button"
            icon="file_download"
          />
        ),
      })
    }

    sheetData.push(macStateAndSettings)

    return (
      <div className={style.overviewInfo}>
        <div>
          <DataSheet data={sheetData} />
        </div>
      </div>
    )
  }

  render() {
    const { device, shouldRedirect } = this.props
    const devIds = device && device.ids
    const otherwise = {
      redirect: '/applications',
      message: m.failedAccessOtherHostDevice,
    }
    return (
      <Require condition={!shouldRedirect} otherwise={otherwise}>
        <Container>
          <IntlHelmet title={sharedMessages.overview} />
          <Row className={style.head}>
            <Col md={12} lg={6}>
              {this.deviceInfo}
            </Col>
            <Col md={12} lg={6} className={style.latestEvents}>
              <DeviceEvents devIds={devIds} widget />
              <DeviceMap device={device} />
            </Col>
          </Row>
        </Container>
      </Require>
    )
  }
}

export default DeviceOverview
