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
import { useSelector } from 'react-redux'
import classnames from 'classnames'
import { defineMessages } from 'react-intl'

import Link from '@ttn-lw/components/link'
import Icon from '@ttn-lw/components/icon'
import Status from '@ttn-lw/components/status'

import DateTime from '@ttn-lw/lib/components/date-time'
import Message from '@ttn-lw/lib/components/message'

import Notification from '@console/components/notifications'

import notificationStyle from '@console/containers/notifications/notifications.styl'

import sharedMessages from '@ttn-lw/lib/shared-messages'

import {
  selectInboxNotifications,
  selectInboxNotificationsTotalCount,
} from '@console/store/selectors/notifications'

import style from './header.styl'

const m = defineMessages({
  description: 'Showing last 3 of {totalNotifications} notifications',
})

const NotificationsDropdown = () => {
  const dropdownItems = useSelector(selectInboxNotifications)
  const totalNotifications = useSelector(selectInboxNotificationsTotalCount)

  return (
    <>
      <div className={style.notificationsDropdownHeader}>
        <span>
          <Message content={sharedMessages.notifications} />{' '}
          <Message
            className="c-text-neutral-semilight fw-normal fs-m"
            content={`(${totalNotifications})`}
          />
        </span>
      </div>
      {dropdownItems.slice(0, 3).map(notification => (
        <Link
          to={{
            pathname: `/notifications/all/${notification.id}`,
          }}
          key={notification.id}
          className={classnames(style.notificationsDropdownLink, 'd-flex')}
        >
          <Icon icon="key" className={style.notificationsDropdownLinkIcon} />
          <div className={style.notificationContainer}>
            <div className={classnames(style.title, 'fw-bold')}>
              <Notification.Title
                data={notification}
                notificationType={notification.notification_type}
              />
            </div>
            <div
              className={classnames(
                notificationStyle.notificationPreviewContent,
                style.previewContent,
              )}
            >
              <Notification.Preview
                data={notification}
                notificationType={notification.notification_type}
              />
            </div>
            <Status pulse={false} status="good" className="mr-cs-xs" />
            <DateTime.Relative
              relativeTimeStyle="short"
              showAbsoluteAfter={2}
              dateTimeProps={{
                time: false,
                dateFormatOptions: { month: '2-digit', day: '2-digit', year: 'numeric' },
              }}
              value={notification.created_at}
              className="fs-s c-text-neutral-heavy"
            />
          </div>
        </Link>
      ))}
      <div className="p-cs-l c-text-neutral-light fs-s text-center c-bg-brand-extralight br-l">
        <Message content={m.description} values={{ totalNotifications }} />
      </div>
    </>
  )
}

export default NotificationsDropdown