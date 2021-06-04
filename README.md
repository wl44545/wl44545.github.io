<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
-->


<br>
<p align="center">
	<a href="https://github.com/wl44545/praca-inzynierska">
		<img src="images/logo.png" alt="Logo" width="80" height="80">
	</a>
	<h2 align="center">Praca inżynierska</h2>
	<h3 align="center">Rozpoznawanie choroby COVID-19 na zdjęciach rentgenowskich płuc z wykorzystaniem
	uczenia maszynowego</h3>
	<h4 align="center">Recognition of COVID-19 disease from X-ray chest images with application of
	machine learning</h4>
</p>
<br>
<br>

<details open="open">
  <summary>Spis treści</summary>
  <ol>
    <li>
      <a href="#szczegóły-pracy-inżynierskiej">Szczegóły pracy inżynierskiej</a>
      <ul>
        <li><a href="#temat-pracy">Temat pracy</a></li>
		<li><a href="#cel-pracy">Cel pracy</a></li>
		<li><a href="#zakres-prac">Zakres prac</a></li>
		<li><a href="#opiekun-pracy">Opiekun pracy</a></li>
		<li><a href="#katedra">Katedra</a></li>
      </ul>
    </li>
	<li>
      <a href="#implementacja">Implementacja</a>
      <ul>
		<li><a href="#planowany-harmonogram">Planowany harmonogram</a></li>
		<li><a href="#postęp-prac">Postęp prac</a></li>
		<li><a href="#biblioteki">Biblioteki</a></li>
      </ul>
    </li>
    <li><a href="#dokumentacja">Dokumentacja</a></li>
	<li><a href="#kontakt">Kontakt</a></li>
  </ol>
</details>

## Szczegóły pracy inżynierskiej

### Temat pracy
<b>Rozpoznawanie choroby COVID-19 na zdjęciach rentgenowskich płuc z wykorzystaniem
uczenia maszynowego</b>
<br>
<i>Recognition of COVID-19 disease from X-ray chest images with application of
machine learning</i>

### Cel pracy
Przeprowadzenie w środowisku programistycznym Python badań porównawczych nad dokładnością rozpoznawania choroby COVID-19 (na podstawie zdjęć rentgenowskich płuc) z wykorzystaniem różnych algorytmów uczenia maszynowego, w szczególności: głębokich sieci neuronowych, boostingu, algorytmu SVM oraz klasyfikatorów bayesowskich.

### Zakres prac
1. Opis wybranych zagadnień związanych z chorobą COVID-19 oraz przedstawienie dostępnego zbioru danych do analizy.
2. Omówienie wybranych algorytmów z zakresu uczenia maszynowgo oraz ekstrakcji cech z obrazów przydatnych do
rozwiązania zadania.
3. Przedstawienie opracowanego oprogramowania.
4. Eksperymenty, przedstawienie otrzymanych miar dokładności (w tym czułość, specyficzność, F1, AUC), porównania,
wnioski końcowe.

### Opiekun pracy
dr hab. inż., prof. ZUT Przemysław Klęsk
<br>
[pklesk@wi.zut.edu.pl](pklesk@wi.zut.edu.pl)

### Katedra
Zachodniopomorski Uniwersytet Technologiczny w Szczecinie
<br>
Wydział Informatyki
<br>
Katedra Sztucznej Inteligencji i Matematyki Stosowanej


## Implementacja

### Planowany harmonogram

A. Część programistyczna
1. Opracowanie zbioru danych
	* zebranie danych
	* skalowanie obrazów
	* przygotowanie danych
2. Implementacja algorytmów
	* Naiwny klasyfikator Bayesa
	* Boosting
		* AdaBoost
		* GradientBoost
	* SVM
		* Liniowe SVM
		* Nieliniowe SVM
	* Głębokie sieci neuronowe
		* VGG-19
		* ResNet-50
		* EfficientNet-B0
		* DenseNet-121
3. Opracowanie miar jakości klasyfikacji
	* Krzywa ROC
	* Tabele kontyngencji
		* wyniki prawdziwie pozytywne
		* wyniki prawdziwie negatywne
		* wyniki fałszywie pozytywne
		* wyniki fałszywie negatywne
	* Miary jakości
		* dokładność
		* prezycja
		* czułość
		* specyficzność
		* F1

B. Część opisowa
1. Wstęp
2. COVID-19
3. Uczenie maszynowe
	* Opis
	* Algorytmy
		* Naiwny klasyfikator Bayesa
		* Boosting
			* AdaBoost
			* GradientBoost
		* SVM
			* Liniowe SVM
			* Nieliniowe SVM
		* Głębokie sieci neuronowe
			* VGG-19
			* ResNet-50
			* EfficientNet-B0
			* DenseNet-121
4. Zbiór danych
	* Źródło
	* Zawartość
	* Opis zdjęć
5. Metodologia
	* Opis
	* Obróbka danych
6. Wyniki
	* Krzywa ROC
	* Tabele kontyngencji
		* wyniki prawdziwie pozytywne
		* wyniki prawdziwie negatywne
		* wyniki fałszywie pozytywne
		* wyniki fałszywie negatywne
	* Miary jakości
		* dokładność
		* prezycja
		* czułość
		* specyficzność
		* F1	
7. Porównanie wyników
8. Wnioski	


### Postęp prac

* BRAK


### Biblioteki
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [NumPy](https://numpy.org/)
* [SciPy](https://www.scipy.org/)
* [OpenCV](https://opencv.org/)


## Dokumentacja
[Dokumentacja](https://github.com/wl44545/praca-inzynierska/documentation)


## Kontakt
Łukasz Więckowski
<br>
[lukasz_wieckowski@zut.edu.pl](lukasz_wieckowski@zut.edu.pl)
<br>
[linkedin.com/lukaszwieckowski](https://www.linkedin.com/in/lukaszwieckowski)



[contributors-shield]: https://img.shields.io/github/contributors/wl44545/praca-inzynierska.svg?style=for-the-badge
[contributors-url]: https://github.com/wl44545/praca-inzynierska/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wl44545/praca-inzynierska.svg?style=for-the-badge
[forks-url]: https://github.com/wl44545/praca-inzynierska/network/members
[stars-shield]: https://img.shields.io/github/stars/wl44545/praca-inzynierska.svg?style=for-the-badge
[stars-url]: https://github.com/wl44545/praca-inzynierska/stargazers
[issues-shield]: https://img.shields.io/github/issues/wl44545/praca-inzynierska.svg?style=for-the-badge
[issues-url]: https://github.com/wl44545/praca-inzynierska/issues
[license-shield]: https://img.shields.io/github/license/wl44545/praca-inzynierska.svg?style=for-the-badge
[license-url]: https://github.com/wl44545/praca-inzynierska/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
