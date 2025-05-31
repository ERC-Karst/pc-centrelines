robust-compare
==============
Compares a pair of non-overlapping increasing sequences.

# Example

```javascript
var robustCompare = require("robust-compare")

console.log(robustCompare([1, 1e64]), robustCompare([-10000, -1e64]))
```

# Install

```
npm install robust-compare
```

# API

#### `require("robust-compare")(a, b)`
Given a pair of non-overlapping increasing sequences, determine if `a` or `b` is smaller/larger.

* `a` and `b` are both non-overlapping increasing sequences

**Returns** A number giving the rank of `a` relative to `b`

* `< 0` if `a < b`
* `= 0` if `a = b`
* `> 0` if `a > b`

# Credits
(c) 2014 Mikola Lysenko. MIT License