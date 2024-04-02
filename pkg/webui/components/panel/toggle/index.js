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

import PropTypes from '@ttn-lw/lib/prop-types'

import styles from './toggle.styl'

const Toggle = ({ options, onToggleChange, active }) => (
  <div className={styles.toggle}>
    {options.map(({ label, value }) => {
      const buttonClassName = classnames(styles.toggleButton, {
        [styles.toggleButtonActive]: value === active,
      })

      return (
        <Button
          key={value}
          message={label}
          value={value}
          onClick={onToggleChange}
          className={buttonClassName}
        />
      )
    })}
  </div>
)

Toggle.propTypes = {
  active: PropTypes.string.isRequired,
  onToggleChange: PropTypes.func.isRequired,
  options: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.message.isRequired,
      value: PropTypes.string.isRequired,
    }),
  ).isRequired,
}

export default Toggle
