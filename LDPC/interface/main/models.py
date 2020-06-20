from django.db import models

# Create your models here.

class Channel(models.Model):
    channel_type = models.CharField(max_length = 200)
    channel_summary = models.CharField(max_length = 200)
    channel_img = models.ImageField(upload_to = "figs")
    channel_slug = models.SlugField(max_length = 40)

    def __str__(self):
        return self.channel_type
    
    