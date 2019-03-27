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
import { connect } from 'react-redux'
import { Switch, Route } from 'react-router'
import { Col, Row, Container } from 'react-grid-system'
import { defineMessages } from 'react-intl'

import sharedMessages from '../../../lib/shared-messages'
import Message from '../../../lib/components/message'
import { withBreadcrumb } from '../../../components/breadcrumbs/context'
import Breadcrumb from '../../../components/breadcrumbs/breadcrumb'
import Spinner from '../../../components/spinner'
import Tabs from '../../../components/tabs'
import IntlHelmet from '../../../lib/components/intl-helmet'

import DeviceOverview from '../device-overview'

import { getDevice } from '../../store/actions/device'

import style from './device.styl'

const m = defineMessages({
  title: '%s - {deviceName} - The Things Network Console',
})

const tabs = [
  { title: 'Overview', name: 'overview' },
  { title: 'Data', name: 'data' },
  { title: 'Location', name: 'location' },
  { title: 'Payload Formatter', name: 'develop' },
  { title: 'General Settings', name: 'general-settings' },
]

@connect(function ({ device, application }, props) {
  return {
    appName: application.application.name,
    deviceName: device.device && device.device.name,
    devId: props.match.params.devId,
    fetching: device.fetching,
    error: device.error,
  }
})
@withBreadcrumb('device.single', function (props) {
  const { appId, devId } = props
  return (
    <Breadcrumb
      path={`/console/applications/${appId}/devices/${devId}`}
      icon="device"
      content={devId}
    />
  )
})
export default class Device extends React.Component {

  componentDidMount () {
    const { dispatch, devId } = this.props

    dispatch(getDevice(devId))
  }

  handleTabChange () {

  }

  render () {
    const { fetching, error, match, devId, deviceName } = this.props

    if (fetching) {
      return (
        <Spinner center>
          <Message content={sharedMessages.loading} />
        </Spinner>
      )
    }

    // show any device fetching error, e.g. not found, no rights, etc
    if (error) {
      return 'ERROR'
    }

    return (
      <React.Fragment>
        <IntlHelmet
          titleTemplate={m.title} values={{ deviceName: deviceName || devId }}
        />
        <Container>
          <Row>
            <Col lg={12}>
              <div className={style.title}>
                <h2 className={style.id}>
                  {devId}
                </h2>
              </div>
              <Tabs
                narrow
                active="overview"
                tabs={tabs}
                onTabChange={this.handleTabChange}
                className={style.tabs}
              />
            </Col>
          </Row>
        </Container>
        <Switch>
          <Route exact path={`${match.path}`} component={DeviceOverview} />
        </Switch>
      </React.Fragment>
    )
  }
}
