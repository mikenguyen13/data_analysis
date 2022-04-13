specify_model <- function(input = NULL, choice = NULL) {
  # This is a template for a design step
  # to be used by the rdfanalysis package
  # See https://joachim-gassen.github.io/rdfanalysis
  # for further information on how to use this template
  # and the package.

  # Feel free to delete any/all lines containing comments but leave
  # the line containing ("Analysis code starts below") unchanged.

  # Provide your documentation in the two variables below.
  # They will be used by prepare_design_documentation()

  step_description <- c(
    "## [Step Title]",
    "### Content",
    "",
    "[Explain what the step does]"
  )
  choice_description <- c(
    "### Choice",
    "",
    "`[name of discrete choice]`: A character value containing one of the",
    "following values:",
    "",
    "- `[valid_value 1]`: [What this choice does]",
    "- `[valid_value n]`: [What that choice does]",
    "",
    "`[name of continuous choice]`: [Explain valid range for continuous choice",
    "and what it does]"
  )

  # Specify your valid choices below. Format will be checked by test_design()
  # for consistency

  choice_type <- list(
    list(name = "[name of discrete choice]",
         type = "character",
         valid_values = c("[valid_value 1]", "[valid_value n]"),
         # weights are not required but they need to add up to one if
         # they are provided. Adjust the example below
         weights = c(0.8, 0.2)),
    list(name = "[name of continuous choice]",
         type = "double",
         valid_min = 0, valid_max = 0.1,
         # weights for continuous choices are provided by a sample of choices.
         # Could be just one value. Adjust the example below
         weight_sample = c(0, rep(0.01, 4), rep(0.05, 4)))
  )

  # You should not need to change anything in the following code section

  if (is.null(choice)) return(list(
    step_description = step_description,
    choice_description = choice_description,
    choice_type = choice_type
  )) else check_choice(choice, choice_type)

  # Leave the following comment untouched. It is being used by
  # prepare_design_documentation() to identify the part of the step
  # that should be included in the documentation.

  # ___ Analysis code starts below ___

  # Here you need to add your code.

  # If this is the first step of your design, you can access input data via the
  # input parameter. If it is a subsequent step, you can access input data
  # via input$data and the protocol leading to this step by input$protocol.

  # Make sure that you address all choices identified above.

  # In the return block below, you need to replace the placeholder with the
  # variable that contains the data that you want to return as output
  # to the next step or as a result if this is the last step.

  protocol <- input$protocol
  protocol[[length(protocol) + 1]] <- choice
  return(list(
    data = "[variable containing your output data structure here]",
    protocol = protocol
  ))
}
