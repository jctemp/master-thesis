#let accent = rgb(220,60,5)


#let project(details, body) = {
  // ==========================================================================
  // SETTINGS

  set document(
    title: details.title,
    author: details.author.name,
  )

  set page(
    paper: "presentation-4-3",
    footer-descent: 10mm,
  )

  set text(
    // font: "Linux Libertine", 
    font: "CMU Serif", 
    size: details.fontSize * 1.75,
    lang: details.language
  )

  show math.equation: set text(weight: 400)
  set math.equation(numbering: "(1.1)")

  show heading.where(level:1): set text(size: 3 * details.fontSize)
  show heading.where(level:2): set text(size: 2 * details.fontSize)
  set heading(numbering: "1.1")

  set cite(style: "alphanumeric", form: "normal")

  set page(
      footer: locate(loc => {
        let page = loc.position().page

        if page > 2 {[
          #show par: set block(spacing: 0.65em)
          #set text(size: details.fontSize * 0.9, weight: "regular")
          #line(length: 100%, stroke: 4pt + accent)
          *Hochschule Hannover* -- #details.author.name -- #details.title 
          #h(1fr) #page
        ]}
      })
    )

  body
}

#let title-slide(details) = {

  image("assets/wordmark_black.png", width: 15%)

  set text(size: details.fontSize * 2.5, weight: "bold")
  details.title
  linebreak()
  linebreak()

  set text(size: details.fontSize * 1.75, weight: "regular")
  details.author.name
  linebreak()
  details.date.display("[day].[month].[year]")

  place(bottom + left, dy: 10mm, image("assets/Medizinische_Hochschule_Hannover_logo.png", width: 80mm))
  place(bottom + right, dy: 10mm, image("assets/logo_colour.png", width: 30mm))

  pagebreak()

  locate(loc => {
    show outline.entry: it => {
      let it-loc = it.element.location()
      if it.level == 1 {
        v(details.fontSize * 3, weak: true)
        box(
          fill: if loc.page() == it-loc.page() { accent } else { white },
          outset: 4pt,
          width: 100%, grid(
          columns: (auto, 1fr, auto),
          link(it-loc, strong(it.body)),
          strong(" " + box(width: 1fr, repeat[.])),
          strong(" " + [#it-loc.page()])
        )
        )
      } else if it.level == 2 {
        box(width: 100%, grid(
          columns: (auto, 1fr, auto),
          gutter: 1em,
          link(it-loc, align(left, it.body)),
          " " + box(width: 1fr, repeat[.]),
          " " + [#it-loc.page()]
        ))
      }
    }

    outline(indent: true)
  })

  counter(page).update(1) 
}

#let section-slide(details, title) = {
  pagebreak()

  counter(heading).step(level: 1)
  heading(title, level: 1, numbering: none)
  
  locate(loc => {
    show outline.entry: it => {
      let it-loc = it.element.location()
      if it.level == 1 {
        v(details.fontSize * 3, weak: true)
        box(
          fill: if loc.page() == it-loc.page() { accent } else { white },
          outset: 4pt,
          width: 100%, grid(
          columns: (auto, 1fr, auto),
          link(it-loc, strong(it.body)),
          strong(" " + box(width: 1fr, repeat[.])),
          strong(" " + [#it-loc.page()])
        )
        )
      } else if it.level == 2 {
        box(width: 100%, grid(
          columns: (auto, 1fr, auto),
          gutter: 1em,
          link(it-loc, align(left, it.body)),
          " " + box(width: 1fr, repeat[.]),
          " " + [#it-loc.page()]
        ))
      }
    }

    outline(title: none, indent: true)
  })
}

#let slide(title, body) = {
  pagebreak()
  [
    == #title
    #body
  ]
  place(bottom, dy: 5mm, grid(
    columns: (1fr, 1fr),
    image("assets/Medizinische_Hochschule_Hannover_logo.png", width: 60mm),
    align(right, image("assets/logo_colour.png", width: 10mm))
  ))
}