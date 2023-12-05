// Copyright © 2023 The Things Network Foundation, The Things Industries B.V.
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

import SideBarContext from '../context'

import AppListSideNavigation from './app-list-side-navigation'
import AppSideNavigation from './app-side-navigation'
import GtwListSideNavigation from './gtw-list-side-navigation'
import GtwSideNavigation from './gtw-side-navigation'
import GeneralSideNavigation from './general-side-navigation'

const SideBarNavigation = () => {
  const { layer } = React.useContext(SideBarContext)

  return (
    <div className="mt-cs-xs">
      {(layer === '/' || layer === '/console') && <GeneralSideNavigation />}
      {layer.includes('/applications/') && <AppSideNavigation />}
      {layer.includes('/applications') && <AppListSideNavigation />}
      {layer.includes('/gateways/') && <GtwSideNavigation />}
      {layer.includes('/gateways') && <GtwListSideNavigation />}
    </div>
  )
}

export default SideBarNavigation