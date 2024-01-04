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

import React, { useCallback, useContext, useRef } from 'react'
import classnames from 'classnames'

import Button from '@ttn-lw/components/button'
import Dropdown from '@ttn-lw/components/dropdown'

import { LanguageContext } from '@ttn-lw/lib/components/with-locale'

import SidebarContext from '@console/containers/side-bar/context'

import sharedMessages from '@ttn-lw/lib/shared-messages'
import PropTypes from '@ttn-lw/lib/prop-types'
import {
  selectDocumentationUrlConfig,
  selectPageStatusBaseUrlConfig,
  selectSupportLinkConfig,
} from '@ttn-lw/lib/selectors/env'

import style from './side-footer.styl'

const supportLink = selectSupportLinkConfig()
const documentationBaseUrl = selectDocumentationUrlConfig()
const statusPageBaseUrl = selectPageStatusBaseUrlConfig()

const LanguageOption = ({ locale, title, currentLocale, onSetLocale }) => {
  const handleSetLocale = useCallback(() => {
    onSetLocale(locale)
  }, [locale, onSetLocale])

  return <Dropdown.Item title={title} action={handleSetLocale} active={locale === currentLocale} />
}

LanguageOption.propTypes = {
  currentLocale: PropTypes.string.isRequired,
  locale: PropTypes.string.isRequired,
  onSetLocale: PropTypes.func.isRequired,
  title: PropTypes.string.isRequired,
}

const SideFooter = () => {
  const { isMinimized } = useContext(SidebarContext)
  const supportButtonRef = useRef(null)
  const clusterButtonRef = useRef(null)

  const clusterDropdownItems = (
    <Dropdown.Item title="Cluster selection" icon="public" path="/cluster" />
  )

  const languageContext = useContext(LanguageContext)
  const { locale, supportedLocales, setLocale } = languageContext || {}

  const handleSetLocale = useCallback(
    locale => {
      setLocale(locale)
    },
    [setLocale],
  )

  const languageItems = supportedLocales
    ? Object.keys(supportedLocales).map(l => (
        <LanguageOption
          locale={l}
          key={l}
          title={supportedLocales[l]}
          currentLocale={locale}
          onSetLocale={handleSetLocale}
        />
      ))
    : null

  const supportDropdownItems = (
    <>
      <Dropdown.Item
        title={sharedMessages.documentation}
        icon="menu_book"
        path={documentationBaseUrl}
        external
      />
      <Dropdown.Item title={sharedMessages.support} icon="support" path={supportLink} external />
      <Dropdown.Item
        title={sharedMessages.statusPage}
        icon="monitor_heart"
        path={statusPageBaseUrl}
        external
      />
      {Boolean(languageContext) && (
        <Dropdown.Item
          title={sharedMessages.language}
          icon="language"
          path="/support"
          submenuItems={languageItems}
          external
        />
      )}
    </>
  )

  const sideFooterClassnames = classnames(
    style.sideFooter,
    'd-flex',
    'j-center',
    'al-center',
    'gap-cs-xs',
    'fs-s',
  )

  return (
    <div className={sideFooterClassnames}>
      <Button
        className={classnames(style.sideFooterButton, style.sideFooterFullWidth)}
        secondary
        icon="support"
        message={!isMinimized ? `v${process.env.VERSION} (${process.env.REVISION})` : undefined}
        dropdownItems={supportDropdownItems}
        noDropdownIcon
        ref={supportButtonRef}
      />
      {!isMinimized && (
        <Button
          className={style.sideFooterButton}
          secondary
          icon="public"
          message="EU1"
          noDropdownIcon
          dropdownItems={clusterDropdownItems}
          dropdownClassName={style.sideFooterClusterDropdown}
          ref={clusterButtonRef}
        />
      )}
    </div>
  )
}

export default SideFooter
