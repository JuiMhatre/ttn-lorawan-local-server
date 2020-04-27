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

import { connect } from 'react-redux'
import { bindActionCreators } from 'redux'
import { replace } from 'connected-react-router'

import attachPromise from '@ttn-lw/lib/store/actions/attach-promise'

import { updateOrganization, deleteOrganization } from '@console/store/actions/organizations'

import {
  selectSelectedOrganization,
  selectSelectedOrganizationId,
} from '@console/store/selectors/organizations'

const mapStateToProps = state => ({
  orgId: selectSelectedOrganizationId(state),
  organization: selectSelectedOrganization(state),
})

const mapDispatchToProps = dispatch => ({
  ...bindActionCreators(
    {
      updateOrganization: attachPromise(updateOrganization),
      deleteOrganization: attachPromise(deleteOrganization),
    },
    dispatch,
  ),
  deleteOrganizationSuccess: () => dispatch(replace(`/organizations`)),
})

export default GeneralSettings =>
  connect(
    mapStateToProps,
    mapDispatchToProps,
  )(GeneralSettings)
