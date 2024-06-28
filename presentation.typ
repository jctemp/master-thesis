#import "templates/presentation.typ": *
#import "metadata.typ": details

#details.insert("language", "de") 
#show: body => project(details, body)

#title-slide(details)

#section-slide(details, "First section")

#slide("First slide", [
  #figure(image("figures/w-msa.png", width: 60%), caption: "Caption")
  #lorem(40)
])

#slide("Second slide", [
  #figure(image("figures/w-msa.png", width: 60%), caption: "Caption")
  #lorem(40)
])

#section-slide(details, "Second section")

#slide("Thrid slide", [
  #figure(image("figures/w-msa.png", width: 60%), caption: "Caption")
  #lorem(40)
])
