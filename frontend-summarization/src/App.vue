<script setup lang="ts">
import {RouterView} from 'vue-router'
import {ref} from "vue";
import {useRouter} from 'vue-router'

const links = ref([
  'Summarization',
  'ONNX',
]);

const router = useRouter()

const goToDashboard = (link: string) => {
  switch (link) {
    case 'Summarization':
      router.push('/')
      break;
    case 'ONNX':
      router.push('/onnx')
      break;
    default:
      router.push('/')
  }
}


</script>


<template>
  <v-app>

    <v-app-bar flat>
      <v-container class="fill-height d-flex align-center">


        <v-btn
            v-for="link in links"
            @click="goToDashboard(link)"
            :key="link"
            variant="text"
        >
          {{ link }}
        </v-btn>

        <v-spacer></v-spacer>

      </v-container>
    </v-app-bar>
    <notifications class="mt-10 mr-16"/>

    <v-main class="bg-grey-lighten-4">

      <RouterView/>

    </v-main>
    <v-footer
        absolute
        class=" text-center d-flex flex-column"
        flat
    >

      <div class="text-caption">
        This project aims to build a Rust app that summarizes text, based on the common task of reading and
        summarizing books among students.
      </div>


      <div class="text-caption">
        {{ new Date().getFullYear() }} —<a href="https://sszzz.me">Shuai Zheng</a>
      </div>
    </v-footer>
  </v-app>
</template>

<style scoped>
/*header {*/
/*  line-height: 1.5;*/
/*  max-height: 100vh;*/
/*}*/

/*.logo {*/
/*  display: block;*/
/*  margin: 0 auto 2rem;*/
/*}*/

/*nav {*/
/*  width: 100%;*/
/*  font-size: 12px;*/
/*  text-align: center;*/
/*  margin-top: 2rem;*/
/*}*/

/*nav a.router-link-exact-active {*/
/*  color: var(--color-text);*/
/*}*/

/*nav a.router-link-exact-active:hover {*/
/*  background-color: transparent;*/
/*}*/

/*nav a {*/
/*  display: inline-block;*/
/*  padding: 0 1rem;*/
/*  border-left: 1px solid var(--color-border);*/
/*}*/

/*nav a:first-of-type {*/
/*  border: 0;*/
/*}*/

/*@media (min-width: 1024px) {*/
/*  header {*/
/*    display: flex;*/
/*    place-items: center;*/
/*    padding-right: calc(var(--section-gap) / 2);*/
/*  }*/

/*  .logo {*/
/*    margin: 0 2rem 0 0;*/
/*  }*/

/*  header .wrapper {*/
/*    display: flex;*/
/*    place-items: flex-start;*/
/*    flex-wrap: wrap;*/
/*  }*/

/*  nav {*/
/*    text-align: left;*/
/*    margin-left: -1rem;*/
/*    font-size: 1rem;*/

/*    padding: 1rem 0;*/
/*    margin-top: 1rem;*/
/*  }*/
/*}*/
</style>
