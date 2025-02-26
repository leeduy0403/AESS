## Installation

1. **Clone the repository:**
	```bash
	git clone <https://github.com/leeduy0403/AESS.git>
	cd API
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

## Database Setup

1. **Apply migrations:**
	```bash
	python manage.py makemigration
	python manage.py migrate
	```

## Running the Server

1. **Start the server:**
	```bash
	python manage.py runserver {PORT}
	```