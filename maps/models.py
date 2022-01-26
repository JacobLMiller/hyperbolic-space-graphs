from django.db import models
from json import dumps

class Task(models.Model):
    creation_date = models.CharField(null=True, max_length=64)
    creation_ip = models.CharField(null=True, max_length=64)

    input_dot = models.TextField()

    dot_rep = models.TextField()
    svg_rep = models.TextField()

    svg_rep0 = models.TextField(null=True)
    svg_rep1 = models.TextField(null=True)
    svg_rep2 = models.TextField(null=True)
    svg_rep3 = models.TextField(null=True)

    status = models.TextField()

    width = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)

    vis_type = models.CharField(max_length=64)
    layout_algorithm = models.CharField(max_length=64)
    iterations = models.CharField(max_length=64)
    opt_alpha = models.CharField(max_length=64)
    hyperbolic_projection = models.CharField(max_length=64)

    color_scheme  = models.CharField(max_length=64)
    convergence  = models.CharField(max_length=64)

    def metadata(self):
        return {
            'id': self.id,
            'status': self.status,
            'width': self.width,
            'height': self.height,
        }

    def json_metadata(self):
        return dumps(self.metadata())

    def description(self):
        desc = ''
        desc += 'Visualization Type: ' + self.vis_type + '\n'
        desc += 'Layout Algorithm: ' + self.layout_algorithm + '\n'
        desc += 'Cluster Algorithm: ' + self.cluster_algorithm + '\n'
        desc += 'Color Scheme: ' + self.color_scheme + '\n'
        desc += 'Semantic Zoom: ' + self.semantic_zoom + '\n'
        desc += 'status: ' + self.status + '\n'

        return desc
