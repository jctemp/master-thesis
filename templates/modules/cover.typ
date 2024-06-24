#let cover(details) = {
  set page(numbering: none)

  image("../assets/wordmark_black.png", width: 20%)

  place(horizon, dy: -20mm,[
    #set text(size: details.fontSize * 1.75, weight: "bold", hyphenate: false)
    #details.title
    #linebreak()

    #set text(size: 1.25 * details.fontSize, weight: "regular")
    #details.author.name
    #linebreak()
    #details.degree's thesis in #details.field
    #linebreak()
    #linebreak()
    #details.date
  ])

  place(bottom + right, dy: 10mm, image("../assets/logo_colour.png", width: 40mm))
}