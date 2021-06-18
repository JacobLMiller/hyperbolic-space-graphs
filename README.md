Hyperbolic realization of graphs
================
This webserver will compute a graph layout in hyperbolic space, using tangent planes as described in the below paper. The graphs are displayed in a Poincare disk.

Video showing the algorithm in action available here: https://youtu.be/CesUj4p7GGM

Kobourov, S., Wampler, K.: Non-Euclidean spring embedders. IEEE Trans. Vis.
Comput. Graph. 11(6), 757–767 (2005). [doi](https://doi.org/10.1109/TVCG.2005.103)


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
