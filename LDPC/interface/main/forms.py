from django import forms

class InputForm(forms.Form):
    snr = forms.CharField(label = "SNR", help_text = "Enter an array of SNR values", required = True)
    img = forms.ImageField(help_text = "Input a gray-scale *.png file")
    