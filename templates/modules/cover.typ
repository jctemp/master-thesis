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
    {
      #let today = details.date
      #let remainder = calc.rem(today.day(), 10)
      #let suffix = "[month repr:long] [year]"

      #today.display(
        if remainder == 1 {
          "[day padding:none]st "
        } else if remainder == 2 {
          "[day padding:none]nd "
        } else if remainder == 3 {
          "[day padding:none]rd "
        } else {
          "[day padding:none]th "
        } + suffix)
      }
    ])

  place(bottom + left, dy: 10mm, image("../assets/Medizinische_Hochschule_Hannover_logo.png", width: 100mm))

  place(bottom + right, dy: 10mm, image("../assets/logo_colour.png", width: 40mm))
}