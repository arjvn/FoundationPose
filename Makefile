APP_NAME ?= foundationpose
TEST_SCENE_DIR ?= avocado_translate_1

build: ## Build the container
	docker build -t ${APP_NAME} .

build-nc: ## Build the container without caching
	docker build --no-cache -t $(APP_NAME) . 

run: ## Run container
	docker run --entrypoint '' --gpus all -it --rm --name="$(APP_NAME)" $(APP_NAME) bash

run-bm: ## Run container with bind mount directory
	docker run --entrypoint '' --gpus all -it --rm -v $(PATH_INPUTS_BINDMOUNT_HOST):$(PATH_INPUTS_BINDMOUNT_CONTAINER) --name="$(APP_NAME)" $(APP_NAME) bash

leap-run:
	make build
	bash run_container.sh test_data/$(TEST_SCENE_DIR)
	
clean-container:
	docker stop $(APP_NAME)
	docker rm $(APP_NAME)