import { p_températures_départementales } from "./températures_départementales.js";

/* 
Promesse produisant les températures minimales/moyennes/maximales mensuelles en 2022
dans le département 49 à partir de la réponse de la promesse `p_températures_départementales` (voir import).
La réponse est un tableau d'objets.
A chacun des 12 mois de 2022 correspond 3 objets dans ce tableau : 
- un pour la température minimale, 
- un pour la température maximale et
- un pour la température moyenne.

Chaque objet à 4 propriétés :
    -- "date" : un objet Date correspondant au 1er jour du mois
    -- "mois" : le nom du mois en français
    -- "température" : un flottant représentant une température
    -- "classe" : l'une des chaînes "Minimales", "Maximales", "Moyennes"

La température minimale (resp. maximale) d'un mois donné sera le minimum (resp. maximum) 
des températures minimales (resp. maximales) quotidiennes relevées pour ce mois.
La température moyenne d'un mois sera la moyenne des températures moyennes 
quotidiennes relevées pour ce mois.
*/
export const p_températures_anjou = new Promise((resolve) => {
  var tempAnjou = p_températures_départementales.then((temperature) => {
    let dictMois2022 = [[], [], [], [], [], [], [], [], [], [], [], []];
    temperature[49].forEach((element) => {
      if (element.date.getFullYear() == 2022) {
        dictMois2022[element.date.getMonth()].push([
          element.tmoy,
          element.tmin,
          element.tmax,
        ]);
      }
    });
    var map1 = dictMois2022.map((mois) => {
      var tempMin = Math.min(...mois.map((jour) => jour[1]));
      var tempMax = Math.max(...mois.map((jour) => jour[2]));
      var tempMoy = parseFloat(
        (mois.reduce((acc, jour) => acc + jour[0], 0) / mois.length).toFixed(1)
      );
      return { tmoy: tempMoy, tmin: tempMin, tmax: tempMax };
    });
    return map1;
  });
  resolve(tempAnjou);
});
