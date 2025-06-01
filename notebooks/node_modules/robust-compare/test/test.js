"use strict"

var tape = require("tape")
var cmp = require("../cmp.js")

tape("robust-compare", function(t) {
  t.equals(cmp([5], [1, 4]), 0)
  t.ok(cmp([1e64], [-1e-100, 1e64]) > 0)
  t.ok(cmp([1e64], [1e-100, 1e64]) < 0)

  t.end()
})