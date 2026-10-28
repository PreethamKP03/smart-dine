// Regression checks for frontend explanations and shared consensus definition.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const elements = {};
const document = {
  addEventListener() {},
  getElementById(id) {
    return elements[id] ||= { textContent: '', innerHTML: '', style: {}, classList: { remove() {} } };
  }
};
const context = vm.createContext({document, window: {location: {origin:'http://localhost'}}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../SmartDine/frontend/app.js'),'utf8'),context);
vm.runInContext(`groupMembers = [
 {name:'A',preferred_cuisines:['Italian','Chinese'],budget:800},
 {name:'B',preferred_cuisines:['Italian'],budget:1000}
]; updateConsensusMeter();`,context);
assert.equal(elements.consensusScoreText.textContent,'50%');
vm.runInContext(`openMemberDeepDive({name:'X',location:'North',cost_for_two:900});`,context);
assert.match(elements.modalBody.innerHTML,/13% over budget/);
assert.match(elements.modalBody.innerHTML,/Dietary compatibility is unverified/);
vm.runInContext(`openMemberDeepDive({name:'X',location:'North',cost_for_two:null});`,context);
assert.match(elements.modalBody.innerHTML,/Cost unknown/);
vm.runInContext('groupMembers = [{preferred_cuisines:[]}]; updateConsensusMeter();',context);
assert.equal(elements.consensusScoreText.textContent,'0%');
console.log('5 frontend regression assertions passed.');
