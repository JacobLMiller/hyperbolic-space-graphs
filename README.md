Hyperbolic realization of graphs
================
Repository accompanying IEEE PacificVis submission.

Explanation video:

For HMDS by SGD, take a look at the SGD folder.

Basic Setup
--------

1. Install the python dependencies listed in [requirements.txt](requirements.txt). Using pip:

        pip install -r requirements.txt

2. Install [graphviz](http://graphviz.org/Download..php)

3. Set up Django settings (optional).
Edit `DATABASES`, `SECRET_KEY`, `ALLOWED_HOSTS` and `ADMINS` in `gmap_web/settings.py`

4. Create Django databases:

        ./manage.py syncdb

5. Run the server:

        ./manage.py runserver

6. Access the map interface at `http://localhost:8000`

License
--------
Code is released under the [MIT License](MIT-LICENSE.txt).
