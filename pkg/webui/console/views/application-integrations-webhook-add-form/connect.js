// Copyright © 2021 The Things Network Foundation, The Things Industries B.V.
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

import attachPromise from '@ttn-lw/lib/store/actions/attach-promise'

import { getWebhookTemplate } from '@console/store/actions/webhook-templates'
import { getWebhook } from '@console/store/actions/webhooks'

import { selectSelectedApplicationId } from '@console/store/selectors/applications'
import { selectWebhookTemplateById } from '@console/store/selectors/webhook-templates'

const mapStateToProps = (state, props) => {
  const templateId = props.match.params.templateId

  return {
    appId: selectSelectedApplicationId(state),
    templateId,
    isCustom: !templateId || templateId === 'custom',
    isSimpleAdd: !templateId,
    webhookTemplate: selectWebhookTemplateById(state, templateId),
  }
}

const mapDispatchToProps = dispatch => ({
  getWebhookTemplate: (templateId, selector) => dispatch(getWebhookTemplate(templateId, selector)),
  getWebhook: (appId, webhookId, selector) =>
    dispatch(attachPromise(getWebhook(appId, webhookId, selector))),
})

export default ApplicationWebhookAddForm =>
  connect(mapStateToProps, mapDispatchToProps)(ApplicationWebhookAddForm)
