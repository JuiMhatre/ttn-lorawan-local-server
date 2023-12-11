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

import Button from '@ttn-lw/components/button'

import { selectApplicationSiteName, selectAssetsRootPath } from '@ttn-lw/lib/selectors/env'

const SideHeader = () => {
  const logo = {
    src: `${selectAssetsRootPath()}/tts-logo.svg`,
    alt: `${selectApplicationSiteName()} Logo`,
  }
  return (
    <div className="d-flex j-between">
      <img {...logo} className="w-50" />
      <Button icon="left_panel_close" naked />
    </div>
  )
}

export default SideHeader
