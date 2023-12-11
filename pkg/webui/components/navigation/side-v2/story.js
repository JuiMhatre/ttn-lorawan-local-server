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

import SidebarContext from '@ttn-lw/containers/side-bar/context'

import SideNavigationItem from './item'

import SideNavigation from '.'

export default {
  title: 'Navigation v2',
  component: SideNavigation,
  decorators: [
    storyFn => (
      <SidebarContext.Provider value={{ isMinimized: false }}>{storyFn()}</SidebarContext.Provider>
    ),
  ],
}

export const _SideNavigation = () => (
  <div style={{ width: '300px', height: '700px' }}>
    <SideNavigation>
      <SideNavigationItem title="Overview" path="/" icon="overview" exact />
      <SideNavigationItem title="Devices" path="/devices" icon="devices" />
      <SideNavigationItem title="Data" path="/data" icon="data" />
      <SideNavigationItem title="Payload Formatters" icon="code">
        <SideNavigationItem title="Uplink" path="/payload-formatters/uplink" icon="uplink" />
        <SideNavigationItem title="Downlink" path="/payload-formatters/downlink" icon="downlink" />
      </SideNavigationItem>
      <SideNavigationItem title="Integrations" icon="integration">
        <SideNavigationItem title="MQTT" path="/integrations/mqtt" icon="extension" />
        <SideNavigationItem title="Webhooks" path="/integrations/webhooks" icon="extension" />
        <SideNavigationItem title="Pub/Subs" path="/integrations/pubsubs" icon="extension" />
      </SideNavigationItem>
      <SideNavigationItem title="Collaborators" path="/collaborators" icon="organization" />
      <SideNavigationItem title="API keys" path="/api-keys" icon="api_keys" />
      <SideNavigationItem
        title="General Settings"
        path="/general-settings"
        icon="general_settings"
      />
    </SideNavigation>
  </div>
)
