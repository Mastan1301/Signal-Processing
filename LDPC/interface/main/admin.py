from django.contrib import admin
from .models import Channel
from django.db import models


# Register your models here.

class ChannelAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Title", {"fields": ["channel_type"]}),
        ("Summary", {"fields": ["channel_summary"]}),
        ("URL", {"fields": ["channel_slug"]}),
        ("Background Image", {"fields": ["channel_img"]}),
    ]

admin.site.register(Channel, ChannelAdmin)