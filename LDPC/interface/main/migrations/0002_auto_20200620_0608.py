# Generated by Django 2.1.5 on 2020-06-20 06:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='channel',
            name='channel_img',
            field=models.ImageField(upload_to='media/figs/'),
        ),
    ]
