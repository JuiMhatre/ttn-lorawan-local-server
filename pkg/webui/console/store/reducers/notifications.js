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

import {
  GET_ARCHIVED_NOTIFICATIONS_SUCCESS,
  GET_INBOX_NOTIFICATIONS_SUCCESS,
  GET_UNSEEN_NOTIFICATIONS_PERIODICALLY_SUCCESS,
  MARK_ALL_AS_SEEN_SUCCESS,
  UPDATE_NOTIFICATION_STATUS_SUCCESS,
} from '@console/store/actions/notifications'

const defaultState = {
  notifications: {
    inbox: { entities: [], totalCount: 0 },
    archived: { entities: [], totalCount: 0 },
    unseen: { entities: [], totalCount: 0 },
  },
  unseenTotalCount: undefined,
}

// Update a range of values in an array by using another array and a start index.
const fillIntoArray = (array, start, values, totalCount) => {
  const newArray = [...array]
  const end = Math.min(start + values.length, totalCount)
  for (let i = start; i < end; i++) {
    newArray[i] = values[i - start]
  }
  return newArray
}

const pageToIndices = (page, limit) => {
  const startIndex = (page - 1) * limit
  const stopIndex = page * limit - 1
  return [startIndex, stopIndex]
}

const notifications = (state = defaultState, { type, payload }) => {
  switch (type) {
    case GET_INBOX_NOTIFICATIONS_SUCCESS:
      return {
        ...state,
        notifications: {
          ...state.notifications,
          inbox: {
            entities: fillIntoArray(
              state.notifications.inbox.entities,
              pageToIndices(payload.page, payload.limit)[0],
              payload.notifications,
              payload.totalCount,
            ),
            totalCount: payload.totalCount,
          },
        },
        unseenTotalCount: payload.unseenTotalCount,
      }
    case GET_ARCHIVED_NOTIFICATIONS_SUCCESS:
      return {
        ...state,
        notifications: {
          ...state.notifications,
          archived: {
            entities: fillIntoArray(
              state.notifications.archived.entities,
              pageToIndices(payload.page, payload.limit)[0],
              payload.notifications,
              payload.totalCount,
            ),
            totalCount: payload.totalCount,
          },
        },
      }
    case GET_UNSEEN_NOTIFICATIONS_PERIODICALLY_SUCCESS:
      return {
        ...state,
        unseenTotalCount: payload.totalCount,
      }
    case UPDATE_NOTIFICATION_STATUS_SUCCESS:
      return {
        ...state,
        unseenTotalCount:
          payload.status === 'NOTIFICATION_STATUS_SEEN' && state.unseenTotalCount > 0
            ? state.unseenTotalCount - payload.ids.length
            : state.unseenTotalCount,
      }
    case MARK_ALL_AS_SEEN_SUCCESS:
      return {
        ...state,
        unseenTotalCount: 0,
      }
    default:
      return state
  }
}

export default notifications