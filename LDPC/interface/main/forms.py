from django import forms

class InputFormAWGN(forms.Form):
    snr = forms.CharField(label = "SNR", max_length=40, widget=forms.TextInput(attrs={'class' : 'form-control', 'placeholder': 'Enter an array of SNR values. For eg: 1 2 3'}))
    img = forms.FileField(label = "Input Image",widget=forms.FileInput(attrs={'class' : 'form-control'}))
    CHOICES= (
                ('1', 'Belief-Propagation (BP)'),
                ('2', 'BP using Min-sum'),
                ('3', 'Gallagher-A'),
                )
    select = forms.CharField(label = "Select the decoder", widget=forms.Select(choices=CHOICES, attrs={'class' : 'form-control btn-primary dropdown-toggle select'}))


class InputFormBSC(forms.Form):
    p = forms.CharField(label = "Bit-flipping probability", max_length=40, widget=forms.TextInput(attrs={'class' : 'form-control', 'placeholder': 'Enter an array of \'p\' values. For eg: 0.1 0.2 0.3'}))
    img = forms.FileField(label = "Input Image",widget=forms.FileInput(attrs={'class' : 'form-control'}))
    CHOICES= (
                ('1', 'Belief-Propagation (BP)'),
                ('2', 'Gallagher-A'),
                )

    select = forms.CharField(label = "Select the decoder", widget=forms.Select(choices=CHOICES, attrs={'class' : 'form-control btn-primary dropdown-toggle select'}))


class InputFormBEC(forms.Form):
    p = forms.CharField(label = "Bit Erasure probability", max_length=40, widget=forms.TextInput(attrs={'class' : 'form-control', 'placeholder': 'Enter an array of \'p\' values. For eg: 0.1 0.2 0.3'}))
    img = forms.FileField(label = "Input Image",widget=forms.FileInput(attrs={'class' : 'form-control'}))        
    