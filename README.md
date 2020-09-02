CNN katodien lajitteluun

- polut katodikuvien kansioihin
  - jokainen luokka jaettu omiin training ja test kansioihin

- haetaan data  ja liitetään niihin oikean luokan label (saadaan kansiosta)
- muodostetaan train_images ja train_labels, test_images ja test_labels
- tehdään halutut augmentoinnit (horizontal_flip, vertical_flip)
- muodostetaan haluttu CNN malli
- treenaus (ilman augmentointia, training datan augmentoinnilla tai molempien datojen augmentoinnilla)
