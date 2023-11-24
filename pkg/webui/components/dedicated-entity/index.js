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
import classnames from 'classnames'

import Button from '@ttn-lw/components/button'

import Message from '@ttn-lw/lib/components/message'

import PropTypes from '@ttn-lw/lib/prop-types'

import style from './dedicated-entity.styl'

const DedicatedEntity = ({ label, icon, className, onClick, 'data-test-id': dataTestId }) => (
  <div className={classnames(className, style.dedicatedEntity)} data-test-id={dataTestId}>
    <Button className={style.button} primary grey icon={icon} onClick={onClick} />
    <hr className={style.divider} />
    <Message content={label} className={style.label} component="p" />
  </div>
)

DedicatedEntity.propTypes = {
  className: PropTypes.string,
  'data-test-id': PropTypes.string,
  icon: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  onClick: PropTypes.func,
}

DedicatedEntity.defaultProps = {
  onClick: () => null,
  className: undefined,
  'data-test-id': 'dedicated-entity',
}

export default DedicatedEntity
