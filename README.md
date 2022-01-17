Hyperbolic realization of graphs
================
Repository accompanying IEEE PacificVis submission. Webserver yet to be hosted.

To see the python implementation of HMDS by SGD, take a look at the SGD folder (which is invoked by the webserver).

Basic Setup
--------

1. Install the python dependencies listed using pip:

        pip install -r requirements.txt

2. Install [graphviz](http://graphviz.org/Download..php)

3. Set up Django settings (optional).
Edit `DATABASES`, `SECRET_KEY`, `ALLOWED_HOSTS` and `ADMINS` in `gmap_web/settings.py`

4. Create Django databases:

        python manage.py makemigrations
        python manage.py migrate

5. Run the server:

        python manage.py runserver

6. Access the interface at `http://localhost:8000`

License
--------
Code is released under the [MIT License](MIT-LICENSE.txt).
