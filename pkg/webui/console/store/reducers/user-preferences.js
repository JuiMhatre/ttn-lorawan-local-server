// Copyright © 2024 The Things Network Foundation, The Things Industries B.V.
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

import { GET_BOOKMARKS_LIST_SUCCESS } from '@console/store/actions/user-preferences'
import { GET_USER_ME_SUCCESS } from '@console/store/actions/logout'

const initialState = {
  bookmarks: {
    bookmarks: [],
    totalCount: 0,
  },
  consoleTheme: 'CONSOLE_THEME_SYSTEM',
  dashboardLayouts: {},
  sortBy: {},
}

const userPreferences = (state = initialState, { type, payload }) => {
  switch (type) {
    case GET_BOOKMARKS_LIST_SUCCESS:
      return {
        ...state,
        bookmarks: payload.entities,
        totalCount: payload.totalCount,
      }
    case GET_USER_ME_SUCCESS:
      return {
        ...state,
        consoleTheme: payload.console_preferences.console_theme || 'CONSOLE_THEME_SYSTEM',
        dashboardLayouts: payload.console_preferences.dashboard_layout || {},
        sortBy: payload.console_preferences.sort_by || {},
      }
    default:
      return state
  }
}

export default userPreferences
