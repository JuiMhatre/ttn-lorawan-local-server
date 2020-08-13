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
import { defineMessages, injectIntl } from 'react-intl'
import { Container, Col, Row } from 'react-grid-system'

import api from '@console/api'

import Button from '@ttn-lw/components/button'
import DataSheet from '@ttn-lw/components/data-sheet'
import Tag from '@ttn-lw/components/tag'
import Spinner from '@ttn-lw/components/spinner'

import Message from '@ttn-lw/lib/components/message'
import DateTime from '@ttn-lw/lib/components/date-time'
import IntlHelmet from '@ttn-lw/lib/components/intl-helmet'
import withRequest from '@ttn-lw/lib/components/with-request'

import GatewayMap from '@console/components/gateway-map'
import EntityTitleSection from '@console/components/entity-title-section'
import KeyValueTag from '@console/components/key-value-tag'

import GatewayEvents from '@console/containers/gateway-events'
import GatewayConnection from '@console/containers/gateway-connection'

import withFeatureRequirement from '@console/lib/components/with-feature-requirement'

import convertStrToDataUri from '@ttn-lw/lib/convert-str-to-data-uri'
import downloadObjectAsFile from '@ttn-lw/lib/download-object-as-file'
import { selectGsConfig } from '@ttn-lw/lib/selectors/env'
import PropTypes from '@ttn-lw/lib/prop-types'
import sharedMessages from '@ttn-lw/lib/shared-messages'

import { mayViewGatewayInfo } from '@console/lib/feature-checks'

const m = defineMessages({
  downloadGlobalConf: 'Download global_conf.json',
  showGlobalConf: 'Show global_conf.json',
  globalConf: 'Global configuration',
})

@injectIntl
@withRequest(({ gtwId, loadData }) => loadData(gtwId), () => false)
@withFeatureRequirement(mayViewGatewayInfo, {
  redirect: '/',
})
export default class GatewayOverview extends React.Component {
  state = {
    globalConf: null,
  }

  static propTypes = {
    apiKeysTotalCount: PropTypes.number,
    collaboratorsTotalCount: PropTypes.number,
    gateway: PropTypes.gateway.isRequired,
    gtwId: PropTypes.string.isRequired,
    mayViewGatewayApiKeys: PropTypes.bool.isRequired,
    mayViewGatewayCollaborators: PropTypes.bool.isRequired,
    statusBarFetching: PropTypes.bool.isRequired,
  }

  static defaultProps = {
    apiKeysTotalCount: undefined,
    collaboratorsTotalCount: undefined,
  }

  @bind
  async getGlobalConf() {
    if (this.state.globalConf) return

    const { gtwId } = this.props
    const globalConf = await api.gateway.getGlobalConf(gtwId)
    this.setState({ globalConf })
  }

  @bind
  async handleDownload() {
    await this.getGlobalConf()
    const globalConfStr = convertStrToDataUri(this.state.globalConf)
    downloadObjectAsFile(globalConfStr, 'global_conf.json')
  }

  render() {
    const {
      gtwId,
      collaboratorsTotalCount,
      apiKeysTotalCount,
      statusBarFetching,
      mayViewGatewayApiKeys,
      mayViewGatewayCollaborators,
      gateway: {
        ids,
        name,
        description,
        created_at,
        updated_at,
        frequency_plan_id,
        gateway_server_address,
      },
    } = this.props

    const sheetData = [
      {
        header: sharedMessages.generalInformation,
        items: [
          {
            key: sharedMessages.gatewayID,
            value: gtwId,
            type: 'code',
            sensitive: false,
          },
          {
            key: sharedMessages.gatewayEUI,
            value: ids.eui,
            type: 'code',
            sensitive: false,
          },
          {
            key: sharedMessages.gatewayDescription,
            value: description || <Message content={sharedMessages.none} />,
          },
          {
            key: sharedMessages.createdAt,
            value: <DateTime value={created_at} />,
          },
          {
            key: sharedMessages.updatedAt,
            value: <DateTime value={updated_at} />,
          },
          {
            key: sharedMessages.gatewayServerAddress,
            value: gateway_server_address,
            type: 'code',
            sensitive: false,
          },
        ],
      },
      {
        header: sharedMessages.lorawanInformation,
        items: [
          {
            key: sharedMessages.frequencyPlan,
            value: frequency_plan_id ? <Tag content={frequency_plan_id} /> : undefined,
          },
          {
            key: m.globalConf,
            value: (
              <Button
                type="button"
                secondary
                onClick={this.handleDownload}
                message={m.downloadGlobalConf}
              />
            ),
          },
        ],
      },
    ]

    return (
      <React.Fragment>
        <EntityTitleSection
          entityId={gtwId}
          entityName={name}
          description={description}
          creationDate={created_at}
        >
          <GatewayConnection gtwId={gtwId} />
          {statusBarFetching ? (
            <Spinner after={0} faded micro inline>
              <Message content={sharedMessages.fetching} />
            </Spinner>
          ) : (
            <React.Fragment>
              {mayViewGatewayCollaborators && (
                <KeyValueTag
                  icon="collaborators"
                  value={collaboratorsTotalCount}
                  keyMessage={sharedMessages.collaboratorCounted}
                />
              )}
              {mayViewGatewayApiKeys && (
                <KeyValueTag
                  icon="api_keys"
                  value={apiKeysTotalCount}
                  keyMessage={sharedMessages.apiKeyCounted}
                />
              )}
            </React.Fragment>
          )}
        </EntityTitleSection>
        <Container>
          <IntlHelmet title={sharedMessages.overview} />
          <Row>
            <Col sm={12} lg={6}>
              <DataSheet data={sheetData} />
            </Col>
            <Col sm={12} lg={6}>
              <GatewayEvents gtwId={gtwId} widget />
              <GatewayMap gtwId={gtwId} gateway={this.props.gateway} />
            </Col>
          </Row>
        </Container>
      </React.Fragment>
    )
  }
}
