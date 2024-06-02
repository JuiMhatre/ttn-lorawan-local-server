tools/bin/mage js:build
tools/bin/mage dev:dbStart
tools/bin/mage dev:initStack
go run ./cmd/ttn-lw-stack -c ./config/stack/ttn-lw-stack.yml start

