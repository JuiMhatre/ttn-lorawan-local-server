# Copyright © 2019 The Things Network Foundation, The Things Industries B.V.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: default
default: init

SHELL = bash

include .mage/mage.make
include .make/dev.make

docs-gen:
	hugo -s ./doc --baseUrl https://thethingsnetwork.github.io/lorawan-stack/$(GIT_TAG)/ -d public/$(GIT_TAG)

docs-server:
	hugo server -s ./doc

docs-deps:
	git submodule --init update doc/themes/hugo-theme-techdoc

docs:
	@rm -f doc/ttn-lw-{stack,cli}/*.{md,1,yaml}
	@$(GO) run ./cmd/ttn-lw-stack gen-man-pages --log.level=error -o doc/ttn-lw-stack
	@$(GO) run ./cmd/ttn-lw-stack gen-md-doc --log.level=error -o doc/ttn-lw-stack
	@$(GO) run ./cmd/ttn-lw-stack gen-yaml-doc --log.level=error -o doc/ttn-lw-stack
	@$(GO) run ./cmd/ttn-lw-cli gen-man-pages --log.level=error -o doc/ttn-lw-cli
	@$(GO) run ./cmd/ttn-lw-cli gen-md-doc --log.level=error -o doc/ttn-lw-cli
	@$(GO) run ./cmd/ttn-lw-cli gen-yaml-doc --log.level=error -o doc/ttn-lw-cli

dev-deps: go.deps js.dev-deps

deps: go.deps sdk.deps sdk.js.build js.deps # NOTE: js.deps needs to be AFTER sdk.js.build

test: go.test js.test sdk.test

quality: go.quality js.quality styl.quality snap.quality

build-all: $(MAGE)
	@GO111MODULE=on $(GO) run github.com/goreleaser/goreleaser --snapshot --skip-publish

clean: go.clean js.clean
	rm -rf dist

translations: messages js.translations
