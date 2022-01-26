Hyperbolic realization of graphs
================
Repository accompanying IEEE PacificVis submission. Webserver to be hosted.

HMDS can be run independent of the webserver to produce static images. Take a look at the SGD folder for more details.

Basic Setup
--------

1. Install the python dependencies listed using pip:

        pip install -r requirements.txt

2. Install [graphviz](http://graphviz.org/Download..php) and [graph-tool](https://graph-tool.skewed.de/)

3. Set up Django settings (optional).
Edit `DATABASES`, `SECRET_KEY`, `ALLOWED_HOSTS` and `ADMINS` in `gmap_web/settings.py`

4. Create Django databases:

        python3 manage.py makemigrations
        python3 manage.py migrate

5. Run the server:

        python manage.py runserver

6. Access the interface at `http://localhost:8000`

License
--------
Code is released under the [MIT License](MIT-LICENSE.txt).

Dependencies:
--------
* `python3`
* [`django 3.2`](https://www.djangoproject.com/)
* [`numpy`](http://www.numpy.org/)
* [`graph-tool`](https://graph-tool.skewed.de/)
* [`numba`](https://numba.pydata.org/)
