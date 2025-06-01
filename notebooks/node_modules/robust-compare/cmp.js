"use strict"

module.exports = robustCompare

var diff = require("robust-subtract")

function robustCompare(a, b) {
  var d = diff(a, b)
  return d[d.length-1]
}